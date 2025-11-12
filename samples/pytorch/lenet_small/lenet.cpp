
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

/// Latency=24256, interval=24256
/// DSP=1, BRAM=0
void forward_node0(
  hls::stream<bool> &v0,
  float v1[6][28][28],
  float v2[6][14][14]
) {	// L6, [0,24256)
  #pragma HLS inline
  v0.read();	// L8, [24254,24254)
  for (int v3 = 0; v3 < 2; v3 += 1) {	// L9, [0,24254), iterCycle=12126, II=12126
    for (int v4 = 0; v4 < 2; v4 += 1) {	// L10, [0,12126), iterCycle=6062, II=6062
      for (int v5 = 0; v5 < 6; v5 += 1) {	// L11, [0,6062), iterCycle=1010, II=1010
        for (int v6 = 0; v6 < 14; v6 += 1) {	// L12, [0,1010), iterCycle=72, II=72
          for (int v7 = 0; v7 < 14; v7 += 1) {	// L13, [0,72), iterCycle=5, II=5
            float v8 = v2[v5][v6][v7];	// L14, [0,2)
            float v9 = (v3 == 0 && v4 == 0) ? (float)-INFINITY : v8;	// L15, [2,2)
            float v10 = v1[v5][((v6 * 2) + v3)][((v7 * 2) + v4)];	// L16, [0,2)
            float v11 = max(v9, v10);	// L17, [2,4)
            v2[v5][v6][v7] = v11;	// L18, [4,5)
          }
        }
      }
    }
  }
}

/// Latency=5318290, interval=5318290
/// DSP=3, BRAM=0
void forward_node1(
  float v12[6][3][5][5],
  float v13[3][32][32],
  float v14[6],
  hls::stream<bool> &v15,
  float v16[6][28][28]
) {	// L26, [0,5318290)
  #pragma HLS inline
  for (int v17 = 0; v17 < 3; v17 += 1) {	// L29, [0,5318288), iterCycle=1772762, II=1772762
    for (int v18 = 0; v18 < 5; v18 += 1) {	// L30, [0,1772762), iterCycle=354552, II=354552
      for (int v19 = 0; v19 < 5; v19 += 1) {	// L31, [0,354552), iterCycle=70910, II=70910
        for (int v20 = 0; v20 < 6; v20 += 1) {	// L32, [0,70910), iterCycle=11818, II=11818
          for (int v21 = 0; v21 < 28; v21 += 1) {	// L33, [0,11818), iterCycle=422, II=422
            for (int v22 = 0; v22 < 28; v22 += 1) {	// L34, [0,422), iterCycle=15, II=15
              float v23 = v14[v20];	// L35, [7,9)
              float v24 = v16[v20][v21][v22];	// L36, [7,9)
              float v25 = (v17 == 0 && v18 == 0 && v19 == 0) ? v23 : v24;	// L37, [9,9)
              float v26 = v13[v17][(v21 + v18)][(v22 + v19)];	// L38, [0,2)
              float v27 = v12[v20][v17][v18][v19];	// L39, [0,2)
              float v28 = v26 * v27;	// L40, [2,9)
              float v29 = v25 + v28;	// L41, [9,12)
              bool v30 = v29 > (float)0.000000;	// L42, [12,14)
              float v31 = v30 ? v29 : (float)0.000000;	// L43, [14,14)
              float v32 = (((-v17) + 2) == 0 && ((-v18) + 4) == 0 && ((-v19) + 4) == 0) ? v31 : v25;	// L44, [14,14)
              v16[v20][v21][v22] = v32;	// L45, [14,15)
            }
          }
        }
      }
    }
  }
  v15.write(true);	// L52, [5318288,5318288)
}

/// This is top function.
/// Latency=1122510, interval=1117210
/// DSP=2, BRAM=0
void forward(
  float v33[3][32][32],
  float v34[6][14][14],
  float v35[6][28][28],
  float v36[6][28][28]
) {	// L55, [0,1122510)
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS dataflow

  #pragma HLS interface m_axi offset=slave port=v36 bundle=axi_3

  #pragma HLS interface m_axi offset=slave port=v35 bundle=axi_2

  #pragma HLS interface m_axi offset=slave port=v34 bundle=axi_1

  #pragma HLS interface m_axi offset=slave port=v33 bundle=axi_0

  float v41[6][3][5][5] = {(float)-0.095117, (float)0.025326, (float)0.008041, (float)-0.013523, (float)-0.092086, (float)-0.105180, (float)0.097178, (float)-0.031915, (float)0.102829, (float)0.019885, (float)0.060810, (float)0.018021, (float)0.086763, (float)0.027914, (float)0.008141, (float)-0.088022, (float)-0.086367, (float)-0.082421, (float)0.038382, (float)-0.113786, (float)-0.015530, (float)0.022551, (float)-0.098563, (float)0.043865, (float)-0.063328, (float)0.115281, (float)0.003908, (float)-0.072226, (float)-0.087612, (float)0.074437, (float)-0.034653, (float)0.032404, (float)0.009853, (float)0.013582, (float)0.098888, (float)-0.111692, (float)0.091119, (float)-0.046056, (float)0.103414, (float)0.002598, (float)0.072216, (float)-0.015052, (float)0.036575, (float)-0.098452, (float)0.020181, (float)-0.001790, (float)0.080510, (float)-0.000847, (float)0.051931, (float)0.060788, (float)0.041833, (float)0.070952, (float)0.035938, (float)-0.046453, (float)-0.049943, (float)0.096650, (float)0.056125, (float)-0.046761, (float)0.061471, (float)-0.106369, (float)0.047738, (float)-0.051278, (float)-0.039590, (float)0.098397, (float)0.010774, (float)-0.111072, (float)-0.075499, (float)0.018520, (float)-0.086122, (float)0.085119, (float)-0.084585, (float)0.040556, (float)0.074926, (float)-0.098274, (float)0.022111, (float)0.032457, (float)0.108718, (float)-0.015932, (float)-0.035638, (float)-0.000953, (float)0.080796, (float)-0.087602, (float)0.076891, (float)0.013763, (float)-0.038507, (float)-0.026562, (float)-0.000709, (float)0.038524, (float)0.108592, (float)-0.102921, (float)0.017586, (float)0.059636, (float)0.114188, (float)-0.021934, (float)0.004562, (float)-0.010872, (float)0.031359, (float)0.042231, (float)-0.064747, (float)0.040445, (float)0.087124, (float)-0.013274, (float)-0.010525, (float)-0.023059, (float)-0.035641, (float)-0.083770, (float)-0.082657, (float)0.002859, (float)-0.111745, (float)-0.080846, (float)-0.061925, (float)0.047885, (float)-0.005303, (float)-0.077015, (float)0.078425, (float)-0.029121, (float)-0.100339, (float)-0.096502, (float)-0.015951, (float)-0.059658, (float)0.097893, (float)0.050983, (float)0.048369, (float)-0.041200, (float)-0.098060, (float)0.106595, (float)-0.069140, (float)0.039442, (float)-0.097851, (float)-0.068388, (float)0.054326, (float)-0.064004, (float)0.090147, (float)0.034149, (float)0.095961, (float)0.047305, (float)0.059518, (float)-0.094754, (float)-0.003978, (float)-0.080639, (float)0.035587, (float)-0.085387, (float)0.095188, (float)0.013267, (float)0.034924, (float)-0.049874, (float)-0.051715, (float)0.114553, (float)0.036056, (float)-0.114433, (float)0.087026, (float)0.038787, (float)-0.091885, (float)0.059142, (float)-0.033492, (float)0.008688, (float)-0.041928, (float)-0.026654, (float)-0.043226, (float)-0.029255, (float)-0.028107, (float)-0.087421, (float)-0.020877, (float)-0.047801, (float)-0.091735, (float)0.037623, (float)0.054828, (float)-0.080971, (float)-0.020474, (float)0.006467, (float)0.077675, (float)-0.010935, (float)0.044806, (float)-0.094629, (float)-0.069438, (float)-0.112809, (float)0.022935, (float)0.082890, (float)-0.059640, (float)-0.065995, (float)-0.086913, (float)0.113130, (float)0.037028, (float)0.004795, (float)0.058898, (float)-0.011302, (float)0.033832, (float)0.075861, (float)0.102248, (float)-0.009005, (float)-0.111473, (float)0.044124, (float)0.020126, (float)-0.008613, (float)0.098219, (float)-0.017908, (float)-0.024155, (float)-0.081656, (float)-0.024254, (float)0.076515, (float)0.061884, (float)0.070053, (float)0.076263, (float)-0.000850, (float)0.030923, (float)-0.115142, (float)-0.054454, (float)-0.086198, (float)-0.006699, (float)-0.017187, (float)0.045283, (float)-0.110329, (float)-0.016527, (float)-0.047612, (float)0.081476, (float)-0.091111, (float)-0.085072, (float)0.059244, (float)-0.065428, (float)0.056850, (float)0.110748, (float)-0.072317, (float)-0.081743, (float)0.082538, (float)0.112228, (float)0.021062, (float)0.065618, (float)-0.101441, (float)0.031362, (float)-0.066638, (float)-0.065464, (float)0.097992, (float)0.070474, (float)0.110297, (float)-0.065030, (float)-0.107032, (float)0.071071, (float)-0.090776, (float)-0.074307, (float)0.079053, (float)0.042112, (float)0.009092, (float)-0.038352, (float)0.084770, (float)0.063591, (float)0.051089, (float)0.113966, (float)0.044806, (float)0.085535, (float)0.061265, (float)-0.079374, (float)-0.017709, (float)0.042499, (float)-0.087929, (float)0.037991, (float)0.103061, (float)0.076309, (float)-0.013593, (float)0.037640, (float)-0.109260, (float)0.114675, (float)0.106110, (float)0.014051, (float)0.113762, (float)-0.043410, (float)0.022250, (float)-0.007324, (float)-0.061327, (float)0.055970, (float)0.038101, (float)0.103917, (float)0.020217, (float)-0.023923, (float)-0.072524, (float)-0.061626, (float)-0.044316, (float)0.102817, (float)-0.069448, (float)0.104989, (float)0.011180, (float)0.071899, (float)-0.075829, (float)-0.008770, (float)0.101270, (float)0.011395, (float)-0.106152, (float)-0.063510, (float)-0.082233, (float)-0.067254, (float)0.055525, (float)-0.052156, (float)0.053129, (float)0.056137, (float)-0.025822, (float)-0.043674, (float)-0.048540, (float)-0.036964, (float)0.021113, (float)0.084090, (float)-0.006418, (float)-0.056887, (float)-0.020226, (float)-0.089151, (float)0.031974, (float)0.019201, (float)0.103496, (float)-0.031551, (float)0.096369, (float)0.113461, (float)-0.073802, (float)0.080820, (float)0.045646, (float)-0.019460, (float)-0.048421, (float)0.090794, (float)0.022745, (float)-0.027041, (float)0.053073, (float)0.088712, (float)0.089102, (float)0.002917, (float)0.092013, (float)-0.003915, (float)-0.028127, (float)-0.011709, (float)-0.045018, (float)-0.106918, (float)0.099273, (float)-0.072390, (float)0.018674, (float)-0.066147, (float)-0.047501, (float)-0.104203, (float)-0.008845, (float)0.103675, (float)0.066858, (float)-0.062226, (float)0.013662, (float)0.072338, (float)-0.010057, (float)-0.091456, (float)-0.021441, (float)-0.022384, (float)-0.022345, (float)0.060657, (float)-0.082728, (float)-0.075415, (float)-0.057447, (float)-0.048515, (float)-0.009734, (float)-0.032519, (float)0.049916, (float)-0.059439, (float)0.044554, (float)0.020607, (float)-0.101684, (float)0.001491, (float)0.055220, (float)-0.010862, (float)0.000926, (float)-0.080552, (float)-0.090133, (float)-0.038192, (float)-0.035148, (float)0.077985, (float)0.024732, (float)-0.006245, (float)0.093216, (float)-0.020825, (float)-0.111334, (float)-0.049418, (float)-0.042509, (float)0.110087, (float)0.064125, (float)-0.078190, (float)-0.075046, (float)0.013693, (float)-0.005877, (float)0.082656, (float)-0.045532, (float)-0.095934, (float)-0.018868, (float)0.023287, (float)0.000105, (float)-0.011724, (float)-0.041580, (float)0.023123, (float)-0.031456, (float)-0.077310, (float)-0.098247, (float)0.057070, (float)-0.048978, (float)-0.078811, (float)0.093003, (float)0.058259, (float)0.090775, (float)0.054168, (float)0.082373, (float)0.024606, (float)-0.066662, (float)-0.011965, (float)0.013206, (float)-0.018122, (float)0.002872, (float)-0.035219, (float)-0.000714, (float)0.061497, (float)0.093519, (float)0.108807, (float)0.095154, (float)-0.023805, (float)0.104936, (float)-0.046236, (float)-0.087566, (float)0.074288, (float)0.003766, (float)0.103393, (float)0.018006, (float)0.004996, (float)-0.013239, (float)-0.100626, (float)-0.060006, (float)0.004019, (float)-0.089229, (float)-0.096470, (float)-0.016747, (float)0.105779, (float)-0.038403, (float)0.044274, (float)-0.112186, (float)0.013562, (float)-0.100170, (float)0.089444, (float)0.055587, (float)0.064977, (float)0.098296, (float)-0.064446, (float)0.069652, (float)-0.077366, (float)0.093509, (float)0.059992, (float)-0.045671, (float)0.029510, (float)0.031076, (float)-0.115404, (float)-0.072264, (float)-0.052145, (float)0.040935, (float)-0.043828, (float)-0.028817};	// L64, [0,0)
  #pragma HLS bind_storage variable=v41 type=ram_t2p impl=bram

  float v42[6] = {(float)0.093006, (float)0.062536, (float)-0.074579, (float)-0.048511, (float)-0.062967, (float)-0.081220};	// L65, [0,0)
  #pragma HLS bind_storage variable=v42 type=ram_t2p impl=bram

  hls::stream<bool> v43;	// L66, [0,0)
  forward_node1(v41, v33, v42, v43, v35);	// L67, [0,1117210)
  forward_node0(v43, v36, v34);	// L68, [1117210,1122508)
}

