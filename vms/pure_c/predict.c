#include <stdio.h>

#include "predict.h"

#ifdef USE_OMPSS
#include <nanos6.h>
#include <assert.h>
#endif


static void features_loop(
        const F_base features[num_features],
        L_base latents[num_samples][num_latent],
        const M_base M[num_samples][num_latent],
        const B_base B[num_samples][num_features][num_latent]
        )
{
    for (int d = 0; d < num_features; d++)
    {
        const F_type feature = features[d];
        for (int s = 0; s < num_samples; s++)
            for (int k = 0; k < num_latent; k++)
            {
                L_type  v;
                if (d==0) v = M[s][k];
                else      v = latents[s][k];
                L_type prod = feature * B[s][d][k];
                latents[s][k] = v + prod;
            }
    }
}

static void proteins_loop(
        P_base predictions[num_proteins],
        L_base latents[num_samples][num_latent],
        const U_base U[num_samples][num_proteins][num_latent]
        )
{
    for (int d = 0; d < num_proteins; d++)
    {
        S_type sum = .0F;
        for (int s = 0; s < num_samples; s++)
            for (int k = 0; k < num_latent; k++)
            {
                S_type prod = latents[s][k] * U[s][d][k];
                sum = sum + prod;
            }

        P_type aggr = sum / num_samples;
        predictions[d] = aggr;
    } // end proteins
}


void predict_one_compound(
        int compound,
        const F_base features[num_features],
        P_base predictions[num_proteins],
        const struct Model *m)
{
    L_base latents[num_samples][num_latent];

    perf_start("predict_one_compound");
    send_features(compound, features);
    features_loop(features, latents, m->M, m->B);
    proteins_loop(predictions, latents, m->U);
    send_predictions(compound, predictions);
    perf_end("predict_one_compound");
}

void predict_compounds(
        int start,
        int num_compounds, 
        const F_flx features,
        P_flx predictions,
        const struct Model *m)
{
    perf_start("predict_compounds");

#ifdef USE_OMPSS
    const int num_tasks_per_node = 10;
    const int num_nodes = nanos6_get_num_cluster_nodes();
    const int num_compounds_per_node = num_compounds / num_nodes;

    for (int node_id = 0; node_id < num_nodes; ++node_id)
    {
        const int i = node_id * num_compounds_per_node;
        const int num_compounds_this_node = (node_id != num_nodes-1) ? num_compounds_per_node : num_compounds-i;

        #pragma oss task weakin(features[i;num_compounds_this_node]) weakin(*m) weakout(predictions[i;num_compounds_this_node]) \
                         node(node_id) label("outer_predict_task")
        {
            const int num_compounds_per_task = num_compounds_this_node / num_tasks_per_node;
            assert(num_compounds_per_task > 0);

            for(int k = i; k < i+num_compounds_this_node; k += num_compounds_per_task)
            {
                const int num_compounds_this_task = MIN(num_compounds_per_task, i+num_compounds_this_node-k);

                #pragma oss task in(features[k;num_compounds_this_task]) in(*m) out(predictions[k;num_compounds_this_task]) \
                                 node(nanos6_cluster_no_offload) label("inner_predict_task")
                for (int i = k; i < k+num_compounds_this_task; ++i)
                    predict_one_compound(i, features[i], predictions[i], m);
            }

            #pragma oss taskwait
        } /* end weak task */
    }

#pragma oss taskwait
#else

    /*  OpenMP / plain impl */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(guided)
#endif
    for (int j=start; j<start+num_compounds; j++)
        predict_one_compound(j, features[j], predictions[j], m);
#endif

    perf_end("predict_compounds");
} // end function
