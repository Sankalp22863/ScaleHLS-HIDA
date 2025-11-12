
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

/// Latency=29, interval=29
/// DSP=1, BRAM=0
void forward_node0(
  float v0[3][3],
  float v1
) {	// L5, [0,29)
  #pragma HLS inline
  v1 = (float)-INFINITY;	// L7, [0,1)
  for (int v2 = 0; v2 < 2; v2 += 1) {	// L8, [1,27), iterCycle=12, II=12
    for (int v3 = 0; v3 < 2; v3 += 1) {	// L9, [0,12), iterCycle=5, II=5
      float v4 = v0[v2][v3];	// L10, [0,2)
      float v5 = v1;	// L11, [1,2)
      float v6 = max(v5, v4);	// L12, [2,4)
      v1 = v6;	// L13, [4,5)
    }
  }
}

/// Latency=580, interval=580
/// DSP=3, BRAM=0
void forward_node1(
  float v7[4][4],
  float v8[2][2],
  float v9[3][3]
) {	// L18, [0,580)
  #pragma HLS inline
  for (int v10 = 0; v10 < 2; v10 += 1) {	// L21, [0,578), iterCycle=288, II=288
    for (int v11 = 0; v11 < 2; v11 += 1) {	// L22, [0,288), iterCycle=143, II=143
      for (int v12 = 0; v12 < 3; v12 += 1) {	// L23, [0,143), iterCycle=47, II=47
        for (int v13 = 0; v13 < 3; v13 += 1) {	// L24, [0,47), iterCycle=15, II=15
          float v14 = v9[v12][v13];	// L25, [7,9)
          float v15 = (v10 == 0 && v11 == 0) ? (float)-0.054379 : v14;	// L26, [9,9)
          float v16 = v7[(v12 + v10)][(v13 + v11)];	// L27, [0,2)
          float v17 = v8[v10][v11];	// L28, [0,2)
          float v18 = v16 * v17;	// L29, [2,9)
          float v19 = v15 + v18;	// L30, [9,12)
          bool v20 = v19 > (float)0.000000;	// L31, [12,14)
          float v21 = v20 ? v19 : (float)0.000000;	// L32, [14,14)
          float v22 = (((-v10) + 1) == 0 && ((-v11) + 1) == 0) ? v21 : v15;	// L33, [14,14)
          v9[v12][v13] = v22;	// L34, [14,15)
        }
      }
    }
  }
}

/// This is top function.
/// Latency=138, interval=119
/// DSP=2, BRAM=1
void forward(
  float v23[4][4],
  float v24
) {	// L41, [0,138)
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS dataflow

  #pragma HLS interface bram storage_type=ram_t2p storage_impl=bram port=v24 bundle=axi_1

  #pragma HLS interface bram storage_type=ram_t2p storage_impl=bram port=v23 bundle=axi_0

  float v27[2][2] = {(float)0.485663, (float)-0.389278, (float)0.109658, (float)-0.429233};	// L46, [0,0)
  #pragma HLS bind_storage variable=v27 type=ram_t2p impl=bram

  float v28[3][3];	// L47, [0,0)
  #pragma HLS bind_storage variable=v28 type=ram_t2p impl=bram

  forward_node1(v23, v27, v28);	// L48, [0,119)
  forward_node0(v28, v24);	// L49, [119,136)
}

