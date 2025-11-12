
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

/// Latency=13612, interval=13612
/// DSP=2, BRAM=1
void forward_node0(
  float v0[84],
  float v1[84][10],
  float v2[10],
  float v3[10]
) {	// L9, [0,13612)
  #pragma HLS inline
  float v4[10];	// L10, [0,0)
  #pragma HLS bind_storage variable=v4 type=ram_t2p impl=bram

  for (int v5 = 0; v5 < 84; v5 += 1) {	// L11, [0,13610), iterCycle=162, II=162
    for (int v6 = 0; v6 < 10; v6 += 1) {	// L12, [0,162), iterCycle=16, II=16
      float v7 = v0[v5];	// L13, [0,2)
      float v8 = v1[v5][v6];	// L14, [0,2)
      float v9 = v4[v6];	// L15, [7,9)
      float v10 = v7 * v8;	// L16, [2,9)
      float v11 = v9 + v10;	// L17, [9,12)
      v4[v6] = v11;	// L18, [15,16)
      float v12 = v2[v6];	// L19, [10,12)
      float v13 = v11 + v12;	// L20, [12,15)
      if (((-v5) + 83) == 0) {	// L21, [15,16)
        v3[v6] = v13;	// L22, [15,16)
      }
    }
  }
}

/// Latency=181684, interval=181684
/// DSP=3, BRAM=1
void forward_node1(
  float v14[84],
  float v15[120][84],
  float v16[120],
  float v17[84]
) {	// L28, [0,181684)
  #pragma HLS inline
  float v18[84];	// L30, [0,0)
  #pragma HLS bind_storage variable=v18 type=ram_t2p impl=bram

  for (int v19 = 0; v19 < 120; v19 += 1) {	// L31, [0,181682), iterCycle=1514, II=1514
    for (int v20 = 0; v20 < 84; v20 += 1) {	// L32, [0,1514), iterCycle=18, II=18
      float v21 = v16[v19];	// L33, [0,2)
      float v22 = v15[v19][v20];	// L34, [0,2)
      float v23 = v18[v20];	// L35, [7,9)
      float v24 = v21 * v22;	// L36, [2,9)
      float v25 = v23 + v24;	// L37, [9,12)
      v18[v20] = v25;	// L38, [17,18)
      float v26 = v14[v20];	// L39, [10,12)
      float v27 = v25 + v26;	// L40, [12,15)
      bool v28 = v27 > (float)0.000000;	// L41, [15,17)
      float v29 = v28 ? v27 : (float)0.000000;	// L42, [17,17)
      if (((-v19) + 119) == 0) {	// L43, [17,18)
        v17[v20] = v29;	// L44, [17,18)
      }
    }
  }
}

/// Latency=864804, interval=864804
/// DSP=3, BRAM=1
void forward_node2(
  float v30[16][5][5],
  float v31[120],
  float v32[400][120],
  float v33[120]
) {	// L50, [0,864804)
  #pragma HLS inline
  float v34[120];	// L52, [0,0)
  #pragma HLS bind_storage variable=v34 type=ram_t2p impl=bram

  for (int v35 = 0; v35 < 400; v35 += 1) {	// L53, [0,864802), iterCycle=2162, II=2162
    for (int v36 = 0; v36 < 120; v36 += 1) {	// L54, [0,2162), iterCycle=18, II=18
      float v37 = v30[(v35 / 25)][((v35 % 25) / 5)][(v35 % 5)];	// L55, [0,2)
      float v38 = v32[v35][v36];	// L56, [0,2)
      float v39 = v34[v36];	// L57, [7,9)
      float v40 = v37 * v38;	// L58, [2,9)
      float v41 = v39 + v40;	// L59, [9,12)
      v34[v36] = v41;	// L60, [17,18)
      float v42 = v31[v36];	// L61, [10,12)
      float v43 = v41 + v42;	// L62, [12,15)
      bool v44 = v43 > (float)0.000000;	// L63, [15,17)
      float v45 = v44 ? v43 : (float)0.000000;	// L64, [17,17)
      if (((-v35) + 399) == 0) {	// L65, [17,18)
        v33[v36] = v45;	// L66, [17,18)
      }
    }
  }
}

/// Latency=8784, interval=8784
/// DSP=1, BRAM=0
void forward_node3(
  hls::stream<bool> &v46,
  float v47[16][10][10],
  float v48[16][5][5]
) {	// L72, [0,8784)
  #pragma HLS inline
  v46.read();	// L73, [8782,8782)
  for (int v49 = 0; v49 < 2; v49 += 1) {	// L74, [0,8782), iterCycle=4390, II=4390
    for (int v50 = 0; v50 < 2; v50 += 1) {	// L75, [0,4390), iterCycle=2194, II=2194
      for (int v51 = 0; v51 < 16; v51 += 1) {	// L76, [0,2194), iterCycle=137, II=137
        for (int v52 = 0; v52 < 5; v52 += 1) {	// L77, [0,137), iterCycle=27, II=27
          for (int v53 = 0; v53 < 5; v53 += 1) {	// L78, [0,27), iterCycle=5, II=5
            float v54 = v47[v51][((v52 * 2) + v49)][((v53 * 2) + v50)];	// L79, [0,2)
            float v55 = v48[v51][v52][v53];	// L80, [0,2)
            float v56 = max(v55, v54);	// L81, [2,4)
            v48[v51][v52][v53] = v56;	// L82, [4,5)
          }
        }
      }
    }
  }
}

/// Latency=3653176, interval=3653176
/// DSP=3, BRAM=0
void forward_node4(
  hls::stream<bool> &v57,
  float v58[6][14][14],
  float v59[16],
  float v60[16][6][5][5],
  hls::stream<bool> &v61,
  float v62[16][10][10]
) {	// L90, [0,3653176)
  #pragma HLS inline
  v57.read();	// L93, [3653174,3653174)
  for (int v63 = 0; v63 < 6; v63 += 1) {	// L94, [0,3653174), iterCycle=608862, II=608862
    for (int v64 = 0; v64 < 5; v64 += 1) {	// L95, [0,608862), iterCycle=121772, II=121772
      for (int v65 = 0; v65 < 5; v65 += 1) {	// L96, [0,121772), iterCycle=24354, II=24354
        for (int v66 = 0; v66 < 16; v66 += 1) {	// L97, [0,24354), iterCycle=1522, II=1522
          for (int v67 = 0; v67 < 10; v67 += 1) {	// L98, [0,1522), iterCycle=152, II=152
            for (int v68 = 0; v68 < 10; v68 += 1) {	// L99, [0,152), iterCycle=15, II=15
              float v69 = v59[v66];	// L100, [7,9)
              float v70 = v62[v66][v67][v68];	// L101, [7,9)
              float v71 = (v63 == 0 && v64 == 0 && v65 == 0) ? v69 : v70;	// L102, [9,9)
              float v72 = v58[v63][(v67 + v64)][(v68 + v65)];	// L103, [0,2)
              float v73 = v60[v66][v63][v64][v65];	// L104, [0,2)
              float v74 = v72 * v73;	// L105, [2,9)
              float v75 = v71 + v74;	// L106, [9,12)
              bool v76 = v75 > (float)0.000000;	// L107, [12,14)
              float v77 = v76 ? v75 : (float)0.000000;	// L108, [14,14)
              float v78 = (((-v63) + 5) == 0 && ((-v64) + 4) == 0 && ((-v65) + 4) == 0) ? v77 : v71;	// L109, [14,14)
              v62[v66][v67][v68] = v78;	// L110, [14,15)
            }
          }
        }
      }
    }
  }
  v61.write(true);	// L117, [3653174,3653174)
}

/// Latency=24256, interval=24256
/// DSP=1, BRAM=0
void forward_node5(
  hls::stream<bool> &v79,
  float v80[6][28][28],
  hls::stream<bool> &v81,
  float v82[6][14][14]
) {	// L120, [0,24256)
  #pragma HLS inline
  v79.read();	// L122, [24254,24254)
  for (int v83 = 0; v83 < 2; v83 += 1) {	// L123, [0,24254), iterCycle=12126, II=12126
    for (int v84 = 0; v84 < 2; v84 += 1) {	// L124, [0,12126), iterCycle=6062, II=6062
      for (int v85 = 0; v85 < 6; v85 += 1) {	// L125, [0,6062), iterCycle=1010, II=1010
        for (int v86 = 0; v86 < 14; v86 += 1) {	// L126, [0,1010), iterCycle=72, II=72
          for (int v87 = 0; v87 < 14; v87 += 1) {	// L127, [0,72), iterCycle=5, II=5
            float v88 = v80[v85][((v86 * 2) + v83)][((v87 * 2) + v84)];	// L128, [0,2)
            float v89 = v82[v85][v86][v87];	// L129, [0,2)
            float v90 = max(v89, v88);	// L130, [2,4)
            v82[v85][v86][v87] = v90;	// L131, [4,5)
          }
        }
      }
    }
  }
  v81.write(true);	// L137, [24254,24254)
}

/// Latency=1336690, interval=1336690
/// DSP=12, BRAM=0
void forward_node6(
  float v91[6],
  float v92[6][3][5][5],
  float v93[3][32][32],
  hls::stream<bool> &v94,
  float v95[6][28][28]
) {	// L140, [0,1336690)
  #pragma HLS inline
  for (int v96 = 0; v96 < 3; v96 += 1) {	// L143, [0,1336688), iterCycle=445562, II=445562
    for (int v97 = 0; v97 < 5; v97 += 1) {	// L144, [0,445562), iterCycle=89112, II=89112
      for (int v98 = 0; v98 < 5; v98 += 1) {	// L145, [0,89112), iterCycle=17822, II=17822
        for (int v99 = 0; v99 < 6; v99 += 1) {	// L146, [0,17822), iterCycle=2970, II=2970
          for (int v100 = 0; v100 < 28; v100 += 2) {	// L147, [0,2970), iterCycle=212, II=212
            for (int v101 = 0; v101 < 28; v101 += 2) {	// L148, [0,212), iterCycle=15, II=15
              float v102 = v91[v99];	// L149, [7,9)
              float v103 = v95[v99][v100][v101];	// L150, [7,9)
              float v104 = (v96 == 0 && v97 == 0 && v98 == 0) ? v102 : v103;	// L151, [9,9)
              float v105 = v93[v96][(v100 + v97)][(v101 + v98)];	// L152, [0,2)
              float v106 = v92[v99][v96][v97][v98];	// L153, [0,2)
              float v107 = v105 * v106;	// L154, [2,9)
              float v108 = v104 + v107;	// L155, [9,12)
              bool v109 = v108 > (float)0.000000;	// L156, [12,14)
              float v110 = v109 ? v108 : (float)0.000000;	// L157, [14,14)
              float v111 = (((-v96) + 2) == 0 && ((-v97) + 4) == 0 && ((-v98) + 4) == 0) ? v110 : v104;	// L158, [14,14)
              v95[v99][v100][v101] = v111;	// L159, [14,15)
              float v112 = v95[v99][v100][(v101 + 1)];	// L160, [7,9)
              float v113 = (v96 == 0 && v97 == 0 && v98 == 0) ? v102 : v112;	// L161, [9,9)
              float v114 = v93[v96][(v100 + v97)][((v98 + v101) + 1)];	// L162, [0,2)
              float v115 = v114 * v106;	// L163, [2,9)
              float v116 = v113 + v115;	// L164, [9,12)
              bool v117 = v116 > (float)0.000000;	// L165, [12,14)
              float v118 = v117 ? v116 : (float)0.000000;	// L166, [14,14)
              float v119 = (((-v96) + 2) == 0 && ((-v97) + 4) == 0 && ((-v98) + 4) == 0) ? v118 : v113;	// L167, [14,14)
              v95[v99][v100][(v101 + 1)] = v119;	// L168, [14,15)
              float v120 = v95[v99][(v100 + 1)][v101];	// L169, [7,9)
              float v121 = (v96 == 0 && v97 == 0 && v98 == 0) ? v102 : v120;	// L170, [9,9)
              float v122 = v93[v96][((v97 + v100) + 1)][(v101 + v98)];	// L171, [0,2)
              float v123 = v122 * v106;	// L172, [2,9)
              float v124 = v121 + v123;	// L173, [9,12)
              bool v125 = v124 > (float)0.000000;	// L174, [12,14)
              float v126 = v125 ? v124 : (float)0.000000;	// L175, [14,14)
              float v127 = (((-v96) + 2) == 0 && ((-v97) + 4) == 0 && ((-v98) + 4) == 0) ? v126 : v121;	// L176, [14,14)
              v95[v99][(v100 + 1)][v101] = v127;	// L177, [14,15)
              float v128 = v95[v99][(v100 + 1)][(v101 + 1)];	// L178, [7,9)
              float v129 = (v96 == 0 && v97 == 0 && v98 == 0) ? v102 : v128;	// L179, [9,9)
              float v130 = v93[v96][((v97 + v100) + 1)][((v98 + v101) + 1)];	// L180, [0,2)
              float v131 = v130 * v106;	// L181, [2,9)
              float v132 = v129 + v131;	// L182, [9,12)
              bool v133 = v132 > (float)0.000000;	// L183, [12,14)
              float v134 = v133 ? v132 : (float)0.000000;	// L184, [14,14)
              float v135 = (((-v96) + 2) == 0 && ((-v97) + 4) == 0 && ((-v98) + 4) == 0) ? v134 : v129;	// L185, [14,14)
              v95[v99][(v100 + 1)][(v101 + 1)] = v135;	// L186, [14,15)
            }
          }
        }
      }
    }
  }
  v94.write(true);	// L193, [1336688,1336688)
}

/// This is top function.
/// Latency=2044666, interval=1146606
/// DSP=7, BRAM=3
void forward(
  float v136[3][32][32],
  float v137[10],
  float v138[16][6][5][5],
  float v139[400][120],
  float v140[120][84],
  float v141[6][28][28],
  float v142[6][28][28],
  float v143[6][14][14],
  float v144[6][14][14],
  float v145[16][10][10],
  float v146[16][10][10]
) {	// L196, [0,2044666)
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS dataflow

  #pragma HLS interface m_axi offset=slave port=v146 bundle=axi_10

  #pragma HLS interface m_axi offset=slave port=v145 bundle=axi_9

  #pragma HLS interface m_axi offset=slave port=v144 bundle=axi_8

  #pragma HLS interface m_axi offset=slave port=v143 bundle=axi_7

  #pragma HLS interface m_axi offset=slave port=v142 bundle=axi_6

  #pragma HLS interface m_axi offset=slave port=v141 bundle=axi_5

  #pragma HLS interface m_axi offset=slave port=v140 bundle=axi_4

  #pragma HLS interface m_axi offset=slave port=v139 bundle=axi_3

  #pragma HLS interface m_axi offset=slave port=v138 bundle=axi_2

  #pragma HLS interface bram storage_type=ram_t2p storage_impl=bram port=v137 bundle=axi_1

  #pragma HLS interface m_axi offset=slave port=v136 bundle=axi_0

  float v158[6][3][5][5] = {(float)0.038836, (float)-0.000123, (float)0.064601, (float)-0.081922, (float)-0.044366, (float)0.107141, (float)-0.091984, (float)0.095899, (float)-0.052744, (float)0.000711, (float)0.057997, (float)-0.002007, (float)0.031266, (float)0.023882, (float)-0.006504, (float)0.007304, (float)0.092629, (float)-0.067311, (float)0.045761, (float)-0.101577, (float)-0.080422, (float)0.003828, (float)-0.054257, (float)-0.085367, (float)0.004065, (float)-0.037064, (float)0.110810, (float)-0.036834, (float)0.047208, (float)-0.007713, (float)-0.062422, (float)-0.055718, (float)0.053756, (float)-0.068278, (float)0.038562, (float)0.056618, (float)0.025932, (float)-0.107303, (float)-0.084700, (float)-0.074248, (float)0.052640, (float)0.048388, (float)-0.065640, (float)-0.069154, (float)0.026951, (float)-0.097658, (float)0.052914, (float)-0.054985, (float)-0.077175, (float)-0.031104, (float)0.052004, (float)0.017769, (float)0.102834, (float)-0.075300, (float)-0.097226, (float)0.021637, (float)0.105646, (float)0.010886, (float)0.049287, (float)-0.049574, (float)-0.080307, (float)0.072440, (float)0.045888, (float)0.103052, (float)-0.062755, (float)-0.041906, (float)-0.026149, (float)0.077263, (float)0.092549, (float)-0.053452, (float)0.028149, (float)-0.011838, (float)0.035954, (float)0.002189, (float)0.011440, (float)-0.072466, (float)0.014047, (float)-0.097767, (float)-0.019905, (float)-0.033969, (float)-0.023207, (float)0.042485, (float)0.001531, (float)0.104022, (float)-0.073450, (float)0.088980, (float)0.001127, (float)-0.048317, (float)-0.058965, (float)-0.109061, (float)-0.006961, (float)0.016002, (float)-0.052493, (float)-0.002369, (float)-0.015977, (float)0.033512, (float)-0.059865, (float)0.113746, (float)-0.103640, (float)-0.114905, (float)0.027635, (float)-0.053091, (float)-0.079774, (float)0.056643, (float)0.064605, (float)0.089611, (float)-0.112280, (float)0.033853, (float)-0.082028, (float)0.069474, (float)-0.023304, (float)-0.099201, (float)-0.026412, (float)-0.095818, (float)0.076804, (float)-0.002096, (float)0.086666, (float)-0.102585, (float)-0.005780, (float)-0.071427, (float)-0.041897, (float)0.012302, (float)-0.038983, (float)-0.046679, (float)0.081760, (float)0.023582, (float)-0.039074, (float)-0.014829, (float)-0.013446, (float)-0.095191, (float)0.003959, (float)0.003587, (float)-0.008226, (float)-0.008286, (float)0.070257, (float)-0.019311, (float)0.008844, (float)0.027540, (float)0.090101, (float)-0.093924, (float)-0.008506, (float)0.050799, (float)-0.082042, (float)0.031109, (float)0.110114, (float)-0.055605, (float)-0.035054, (float)-0.046408, (float)-0.060696, (float)0.075218, (float)-0.091260, (float)-0.058770, (float)0.091762, (float)-0.089368, (float)0.060687, (float)-0.005354, (float)-0.028994, (float)-0.012972, (float)-0.036132, (float)0.064982, (float)-0.004808, (float)0.060437, (float)0.115303, (float)-0.015444, (float)0.020923, (float)-0.075119, (float)-0.071234, (float)0.056816, (float)0.102013, (float)-0.006025, (float)-0.098999, (float)0.025401, (float)0.087280, (float)-0.084947, (float)-0.109784, (float)-0.062728, (float)-0.040286, (float)0.078090, (float)0.092630, (float)0.015470, (float)0.105141, (float)-0.074626, (float)-0.063556, (float)0.086452, (float)-0.049530, (float)-0.046895, (float)-0.107669, (float)-0.043818, (float)0.050229, (float)-0.073735, (float)0.090470, (float)-0.005806, (float)0.113636, (float)0.018598, (float)0.060089, (float)0.083202, (float)0.041961, (float)0.047241, (float)-0.041783, (float)0.110657, (float)-0.035858, (float)-0.051497, (float)0.103785, (float)-0.041723, (float)-0.089662, (float)-0.006412, (float)0.006269, (float)-0.053370, (float)0.093155, (float)0.033179, (float)0.030830, (float)-0.101223, (float)0.011994, (float)-0.049724, (float)-0.007697, (float)-0.106928, (float)-0.010795, (float)0.082242, (float)0.020441, (float)0.056803, (float)0.081822, (float)-0.104423, (float)-0.112892, (float)0.096172, (float)0.052234, (float)-0.056987, (float)-0.106818, (float)0.108621, (float)0.075598, (float)-0.053138, (float)-0.085357, (float)-0.010878, (float)-0.037887, (float)-0.025096, (float)-0.002645, (float)0.000530, (float)0.013996, (float)-0.080169, (float)-0.076931, (float)0.040424, (float)-0.043067, (float)-0.019071, (float)0.105925, (float)-0.050022, (float)0.061085, (float)-0.024584, (float)0.033097, (float)0.037475, (float)-0.068287, (float)0.086447, (float)0.082927, (float)0.050917, (float)-0.045578, (float)0.057489, (float)0.044026, (float)0.099334, (float)0.091856, (float)-0.039611, (float)0.081124, (float)-0.049274, (float)-0.100660, (float)0.002890, (float)-0.041508, (float)-0.044995, (float)-0.034392, (float)-0.114542, (float)-0.060810, (float)0.095459, (float)0.004796, (float)-0.000544, (float)-0.095137, (float)0.059426, (float)0.060996, (float)0.072424, (float)0.037675, (float)0.099170, (float)0.021126, (float)-0.098175, (float)0.001113, (float)-0.065048, (float)0.093549, (float)-0.073094, (float)-0.077702, (float)0.091703, (float)0.057270, (float)0.046866, (float)0.028824, (float)0.029263, (float)-0.092757, (float)-0.029838, (float)0.014185, (float)-0.088708, (float)-0.021611, (float)0.007620, (float)-0.056726, (float)-0.021611, (float)0.043881, (float)0.014551, (float)0.090850, (float)0.053056, (float)0.086706, (float)0.111725, (float)0.010761, (float)-0.040689, (float)-0.027812, (float)0.048659, (float)0.061661, (float)0.109287, (float)0.005253, (float)0.087095, (float)0.114719, (float)0.000410, (float)0.048221, (float)0.099112, (float)0.032685, (float)0.020014, (float)-0.010450, (float)-0.001894, (float)-0.063354, (float)0.040561, (float)-0.099860, (float)0.021290, (float)-0.043536, (float)0.040491, (float)0.090290, (float)-0.109607, (float)-0.055675, (float)0.084291, (float)0.045218, (float)-0.044478, (float)-0.108190, (float)0.100047, (float)0.061830, (float)-0.061407, (float)-0.052009, (float)-0.045109, (float)0.038266, (float)-0.090292, (float)0.077673, (float)-0.089383, (float)-0.033395, (float)-0.048896, (float)0.000949, (float)0.100493, (float)-0.047006, (float)0.023067, (float)-0.069872, (float)-0.057688, (float)-0.021081, (float)-0.073393, (float)0.066550, (float)-0.111796, (float)0.020175, (float)-0.027799, (float)-0.085034, (float)-0.057512, (float)0.070903, (float)0.100537, (float)-0.050583, (float)0.024280, (float)0.035349, (float)0.100132, (float)0.053241, (float)-0.067303, (float)0.052259, (float)0.048175, (float)-0.081208, (float)-0.038716, (float)-0.022806, (float)-0.029520, (float)-0.038489, (float)-0.008713, (float)0.090505, (float)-0.096053, (float)-0.093595, (float)0.095422, (float)0.102780, (float)0.074073, (float)0.029514, (float)0.033003, (float)-0.084772, (float)0.104420, (float)-0.062150, (float)0.076155, (float)-0.004726, (float)0.077738, (float)-0.008202, (float)-0.074854, (float)0.055124, (float)-0.062721, (float)0.096692, (float)0.074194, (float)-0.062181, (float)-0.036044, (float)-0.032344, (float)0.040790, (float)0.037025, (float)0.089930, (float)-0.041207, (float)0.032704, (float)-0.065087, (float)0.041350, (float)0.074887, (float)0.081711, (float)-0.113784, (float)0.017130, (float)-0.102078, (float)-0.048441, (float)0.023222, (float)-0.084545, (float)-0.046244, (float)-0.068709, (float)0.110459, (float)-0.067948, (float)0.027223, (float)-0.009795, (float)0.007648, (float)0.085909, (float)-0.001128, (float)-0.091985, (float)0.094112, (float)0.086009, (float)-0.038161, (float)0.017340, (float)0.069843, (float)-0.046024, (float)0.032013, (float)0.092842, (float)-0.020908, (float)-0.074323, (float)-0.041695, (float)-0.020090, (float)-0.032345, (float)0.053416, (float)0.087930, (float)0.005317, (float)-0.081203, (float)0.114508, (float)0.014481, (float)0.083018, (float)-0.088002, (float)0.018587, (float)-0.074058, (float)0.054075, (float)-0.077520, (float)-0.105048, (float)-0.083821, (float)-0.094989, (float)-0.024584, (float)0.113600};	// L219, [0,0)
  #pragma HLS bind_storage variable=v158 type=ram_t2p impl=bram

  float v159[6] = {(float)-0.062924, (float)-0.076287, (float)0.087513, (float)-0.053468, (float)-0.064103, (float)-0.097392};	// L220, [0,0)
  #pragma HLS bind_storage variable=v159 type=ram_t2p impl=bram

  float v160[16] = {(float)-0.040080, (float)0.059602, (float)-0.006795, (float)-0.053933, (float)0.054825, (float)-0.056942, (float)0.053293, (float)0.013058, (float)-0.031937, (float)0.003860, (float)-0.003551, (float)-0.070468, (float)0.041493, (float)0.028134, (float)-0.068392, (float)0.059106};	// L221, [1151904,1151904)
  #pragma HLS bind_storage variable=v160 type=ram_t2p impl=bram

  float v161[120] = {(float)0.049001, (float)-0.009519, (float)0.023160, (float)0.025162, (float)-0.020733, (float)-0.024890, (float)0.009907, (float)-0.047804, (float)0.005592, (float)0.034463, (float)-0.041836, (float)-0.045513, (float)0.036530, (float)-0.022768, (float)-0.047727, (float)0.014279, (float)-0.021068, (float)-0.044461, (float)0.021617, (float)0.003975, (float)-0.000844, (float)-0.026248, (float)-0.031581, (float)0.034957, (float)-0.026718, (float)-0.037853, (float)-0.046846, (float)0.013262, (float)0.046005, (float)-0.011214, (float)-0.003789, (float)-0.039967, (float)-0.042471, (float)-0.027596, (float)0.018269, (float)-0.027379, (float)-0.033587, (float)0.042507, (float)0.017303, (float)0.022775, (float)0.034649, (float)-0.004799, (float)-0.042934, (float)0.046827, (float)-0.043648, (float)-0.005946, (float)0.031980, (float)0.044808, (float)-0.016936, (float)0.045861, (float)0.026385, (float)0.000229, (float)0.037346, (float)-0.035094, (float)0.030034, (float)-0.038585, (float)-0.023800, (float)-0.007012, (float)-0.000567, (float)0.011256, (float)-0.042816, (float)0.016766, (float)-0.024342, (float)0.023685, (float)-0.039887, (float)-0.026746, (float)0.022724, (float)0.013338, (float)0.047056, (float)-0.038719, (float)0.038155, (float)-0.030054, (float)0.045617, (float)0.013677, (float)0.048068, (float)0.021611, (float)-0.049960, (float)0.037242, (float)-0.000763, (float)-0.039694, (float)-0.024703, (float)-0.029220, (float)0.019627, (float)0.025172, (float)-0.001732, (float)-0.015191, (float)0.038678, (float)0.039794, (float)0.008211, (float)0.032198, (float)0.040989, (float)0.006459, (float)-0.019260, (float)0.005920, (float)-0.036001, (float)-0.017981, (float)-0.029254, (float)0.045015, (float)-0.048880, (float)0.019087, (float)-0.028053, (float)-0.013070, (float)-0.035326, (float)0.048529, (float)0.016037, (float)-0.004300, (float)-0.012167, (float)0.008619, (float)0.003753, (float)0.002974, (float)-0.002812, (float)0.013706, (float)0.047608, (float)-0.045835, (float)0.004893, (float)0.033569, (float)0.039309, (float)0.035817, (float)-0.023647, (float)0.001158};	// L222, [1881783,1881783)
  #pragma HLS bind_storage variable=v161 type=ram_t2p impl=bram

  float v162[84] = {(float)-0.021518, (float)-0.051207, (float)0.036601, (float)0.007593, (float)0.045567, (float)-0.062826, (float)0.055591, (float)-0.031913, (float)-0.032279, (float)0.005383, (float)0.047440, (float)-0.037588, (float)-0.055519, (float)0.089972, (float)-0.016793, (float)-0.072175, (float)0.025413, (float)0.007037, (float)-0.044467, (float)0.076578, (float)-0.080474, (float)0.055025, (float)-0.006869, (float)0.035614, (float)0.020357, (float)0.062057, (float)-0.023806, (float)0.078051, (float)0.024315, (float)-0.012002, (float)0.041338, (float)0.030004, (float)-0.070009, (float)-0.081025, (float)0.023246, (float)-0.018533, (float)-0.046368, (float)-0.040512, (float)0.033480, (float)-0.043817, (float)-0.009846, (float)0.081433, (float)-0.078502, (float)0.087739, (float)-0.019978, (float)0.066336, (float)0.063015, (float)0.037246, (float)0.085926, (float)0.064383, (float)-0.036981, (float)-0.044920, (float)0.015959, (float)0.072253, (float)0.055724, (float)0.028237, (float)0.031587, (float)-0.015492, (float)-0.079840, (float)0.002054, (float)-0.025187, (float)0.035419, (float)-0.035609, (float)0.080998, (float)-0.083333, (float)0.052412, (float)0.064371, (float)0.036016, (float)-0.044012, (float)0.025164, (float)0.023746, (float)-0.080663, (float)0.066523, (float)-0.009274, (float)0.020023, (float)0.082046, (float)-0.061348, (float)0.018898, (float)0.077652, (float)-0.056163, (float)-0.080477, (float)0.075232, (float)0.052339, (float)0.068128};	// L223, [2016189,2016189)
  #pragma HLS bind_storage variable=v162 type=ram_t2p impl=bram

  float v163[10] = {(float)-0.008715, (float)-0.070992, (float)0.106052, (float)0.101737, (float)-0.093469, (float)-0.093642, (float)0.077181, (float)-0.015494, (float)0.010494, (float)-0.069723};	// L224, [2042659,2042659)
  #pragma HLS bind_storage variable=v163 type=ram_t2p impl=bram

  float v164[84][10] = {(float)0.092432, (float)-0.041326, (float)0.021496, (float)-0.061636, (float)-0.073410, (float)-0.012221, (float)-0.096583, (float)-0.043975, (float)0.007412, (float)-0.086636, (float)0.075235, (float)0.047429, (float)0.058560, (float)0.053239, (float)-0.018214, (float)0.101287, (float)-0.005639, (float)-0.005061, (float)-0.035847, (float)-0.040949, (float)-0.068141, (float)0.038047, (float)0.053319, (float)0.012827, (float)0.040476, (float)-0.007840, (float)0.051902, (float)-0.067685, (float)-0.019672, (float)0.036655, (float)0.000744, (float)0.066059, (float)-0.037411, (float)-0.069699, (float)-0.042518, (float)-0.072161, (float)0.086445, (float)-0.102408, (float)-0.094029, (float)0.000531, (float)0.002482, (float)-0.068702, (float)-0.066149, (float)0.061407, (float)0.001259, (float)-0.034867, (float)0.026313, (float)0.070866, (float)0.041768, (float)0.056841, (float)-0.069585, (float)-0.049277, (float)0.100024, (float)0.036265, (float)0.051213, (float)0.054840, (float)-0.059105, (float)0.103855, (float)-0.083765, (float)0.077100, (float)0.025119, (float)0.086987, (float)0.038610, (float)0.069694, (float)-0.088608, (float)-0.022709, (float)-0.086163, (float)0.058388, (float)-0.093033, (float)0.015794, (float)0.090712, (float)0.105438, (float)0.003718, (float)-0.069215, (float)-0.009933, (float)-0.097720, (float)0.006759, (float)-0.032848, (float)0.018050, (float)-0.101272, (float)0.029327, (float)0.044022, (float)0.004427, (float)-0.091369, (float)-0.013294, (float)0.040306, (float)-0.039546, (float)0.040788, (float)-0.024522, (float)0.071375, (float)-0.054229, (float)-0.020735, (float)0.051086, (float)0.095242, (float)0.043631, (float)-0.087328, (float)0.099032, (float)-0.073905, (float)-0.093958, (float)0.061682, (float)-0.108871, (float)-0.053465, (float)-0.028738, (float)-0.036343, (float)-0.072962, (float)0.062626, (float)0.103482, (float)0.088295, (float)-0.096682, (float)0.001734, (float)0.092770, (float)0.011373, (float)-0.104942, (float)0.100658, (float)-0.076706, (float)0.030701, (float)0.087495, (float)0.051881, (float)-0.063004, (float)-0.017333, (float)0.092889, (float)0.083097, (float)0.000158, (float)-0.008833, (float)0.090587, (float)-0.021845, (float)-0.097224, (float)0.025208, (float)-0.017168, (float)0.079025, (float)0.010814, (float)0.105558, (float)-0.076562, (float)-0.015255, (float)0.072002, (float)-0.042807, (float)0.084080, (float)0.093954, (float)-0.049691, (float)-0.049594, (float)0.042639, (float)0.050775, (float)-0.002230, (float)0.088922, (float)-0.071236, (float)0.034449, (float)0.101700, (float)0.079641, (float)-0.080646, (float)0.083398, (float)-0.026710, (float)-0.033916, (float)0.032075, (float)0.074636, (float)-0.084617, (float)0.031511, (float)-0.028447, (float)-0.097421, (float)0.063712, (float)0.072330, (float)0.051662, (float)-0.105705, (float)-0.052022, (float)-0.071575, (float)0.061055, (float)-0.106148, (float)-0.044376, (float)-0.012517, (float)-0.003515, (float)0.046900, (float)-0.062704, (float)-0.024784, (float)0.075075, (float)-0.048351, (float)0.073725, (float)0.008243, (float)0.083458, (float)-0.076898, (float)0.031152, (float)-0.093098, (float)0.001351, (float)0.084274, (float)0.074925, (float)-0.077797, (float)-0.077898, (float)0.065638, (float)0.006786, (float)0.095586, (float)-0.003348, (float)0.090063, (float)-0.104335, (float)-0.025313, (float)-0.087471, (float)-0.080849, (float)0.017622, (float)0.072627, (float)-0.001483, (float)-0.053972, (float)0.035731, (float)-0.035693, (float)0.071694, (float)-0.000461, (float)0.108623, (float)-0.052585, (float)-0.085658, (float)0.022250, (float)-0.083983, (float)-0.043310, (float)-0.047660, (float)0.104926, (float)0.014068, (float)0.046141, (float)0.048791, (float)0.082321, (float)0.074627, (float)0.064144, (float)-0.076572, (float)0.026500, (float)0.001593, (float)-0.059484, (float)-0.051027, (float)0.088622, (float)0.074689, (float)0.011137, (float)-0.032910, (float)0.039667, (float)0.027250, (float)0.108574, (float)-0.040940, (float)-0.000489, (float)0.094218, (float)0.059264, (float)-0.028059, (float)0.069468, (float)-0.019251, (float)0.003403, (float)-0.104926, (float)0.055667, (float)0.083649, (float)0.059201, (float)0.053677, (float)0.078922, (float)-0.088999, (float)0.027299, (float)0.044530, (float)0.054397, (float)-0.039294, (float)-0.014682, (float)-0.057752, (float)-0.008155, (float)0.084250, (float)-0.035914, (float)0.068200, (float)-0.009840, (float)0.054761, (float)-0.049512, (float)-0.017001, (float)-0.090626, (float)0.051067, (float)-0.093393, (float)-0.083653, (float)-0.016455, (float)0.000034, (float)0.044221, (float)0.065921, (float)0.086813, (float)-0.076256, (float)-0.051609, (float)-0.073568, (float)-0.058323, (float)0.094307, (float)-0.012708, (float)-0.105805, (float)-0.062231, (float)-0.079439, (float)0.072987, (float)0.007137, (float)0.048320, (float)0.092809, (float)0.033986, (float)0.016948, (float)-0.022126, (float)0.045211, (float)0.075719, (float)-0.010691, (float)-0.045397, (float)0.097391, (float)-0.100415, (float)0.052545, (float)0.073720, (float)0.018290, (float)0.095814, (float)-0.017565, (float)-0.094136, (float)0.045305, (float)0.017231, (float)-0.091583, (float)0.014452, (float)0.102142, (float)0.032911, (float)-0.020145, (float)-0.002442, (float)-0.018141, (float)-0.092906, (float)-0.094764, (float)-0.062677, (float)0.103227, (float)0.057397, (float)0.062926, (float)0.100914, (float)0.106343, (float)0.087397, (float)0.058559, (float)-0.105174, (float)0.043368, (float)-0.095969, (float)0.072710, (float)0.064600, (float)-0.013958, (float)0.081460, (float)0.038423, (float)-0.086786, (float)-0.031529, (float)-0.016849, (float)0.005658, (float)-0.102879, (float)0.006732, (float)-0.090157, (float)0.089705, (float)0.104569, (float)-0.070894, (float)-0.016334, (float)-0.092189, (float)-0.011080, (float)-0.024502, (float)0.011628, (float)-0.068500, (float)0.100132, (float)0.036909, (float)-0.056128, (float)-0.102741, (float)0.031514, (float)0.045398, (float)0.081063, (float)0.107202, (float)0.002982, (float)0.082195, (float)-0.027168, (float)0.028532, (float)0.010358, (float)0.054939, (float)-0.023310, (float)-0.046803, (float)0.013226, (float)0.081444, (float)0.021735, (float)-0.005915, (float)-0.101716, (float)-0.086245, (float)-0.101977, (float)0.012618, (float)0.079422, (float)0.030249, (float)-0.006466, (float)0.047006, (float)0.007179, (float)-0.041633, (float)0.060484, (float)-0.076259, (float)-0.096938, (float)-0.010764, (float)-0.052867, (float)-0.108008, (float)-0.012213, (float)-0.093642, (float)-0.013361, (float)0.084103, (float)-0.097441, (float)0.089059, (float)0.081234, (float)-0.061326, (float)0.007241, (float)-0.068786, (float)0.039684, (float)0.080737, (float)0.098655, (float)-0.049568, (float)-0.051832, (float)0.057538, (float)-0.030586, (float)0.000561, (float)-0.021736, (float)0.098638, (float)-0.058511, (float)0.041375, (float)0.014930, (float)-0.048504, (float)0.002698, (float)0.077455, (float)0.029179, (float)-0.011320, (float)-0.033917, (float)-0.097358, (float)0.079731, (float)-0.075083, (float)0.084491, (float)0.029677, (float)0.063472, (float)-0.056801, (float)0.044590, (float)-0.000178, (float)0.059415, (float)-0.044773, (float)-0.045241, (float)0.058827, (float)0.069661, (float)0.003390, (float)0.008464, (float)0.031450, (float)0.019000, (float)-0.097481, (float)0.105512, (float)0.033408, (float)0.060966, (float)-0.002407, (float)0.004887, (float)0.010570, (float)0.023403, (float)0.084912, (float)0.049694, (float)-0.052558, (float)-0.081284, (float)0.090214, (float)0.007841, (float)-0.039538, (float)0.101007, (float)-0.008787, (float)-0.063216, (float)0.046904, (float)0.064955, (float)-0.031531, (float)-0.042958, (float)-0.050885, (float)0.058352, (float)0.109020, (float)-0.108784, (float)-0.016951, (float)0.074109, (float)0.092273, (float)0.099911, (float)-0.101712, (float)0.080426, (float)-0.018025, (float)-0.074161, (float)-0.085328, (float)-0.099828, (float)0.024599, (float)0.031420, (float)-0.042978, (float)0.012986, (float)0.088160, (float)0.077744, (float)-0.015391, (float)-0.067803, (float)0.103413, (float)0.077876, (float)0.096704, (float)0.000097, (float)0.082137, (float)0.064404, (float)-0.093304, (float)0.098647, (float)-0.075656, (float)0.067672, (float)-0.038327, (float)0.039710, (float)0.104476, (float)-0.057838, (float)-0.102583, (float)-0.001913, (float)0.043858, (float)-0.035503, (float)0.084154, (float)-0.011789, (float)0.073535, (float)0.004609, (float)-0.085284, (float)-0.055127, (float)0.009815, (float)0.000757, (float)0.050786, (float)0.024829, (float)0.086030, (float)0.006946, (float)0.035635, (float)0.085244, (float)-0.103241, (float)-0.080727, (float)-0.088037, (float)0.080324, (float)0.058802, (float)0.053038, (float)0.095971, (float)-0.078211, (float)-0.091858, (float)0.046346, (float)0.004636, (float)0.096076, (float)0.079256, (float)0.020751, (float)0.015666, (float)0.064333, (float)0.007173, (float)-0.096716, (float)-0.077414, (float)0.085756, (float)-0.024687, (float)-0.037322, (float)-0.012335, (float)-0.103299, (float)-0.077919, (float)-0.030819, (float)-0.073159, (float)-0.097737, (float)0.057150, (float)0.008070, (float)-0.066157, (float)0.083575, (float)-0.032456, (float)0.049711, (float)0.093134, (float)0.057135, (float)-0.066237, (float)-0.045046, (float)-0.066783, (float)-0.010538, (float)0.063453, (float)-0.022168, (float)0.064078, (float)0.047437, (float)-0.058178, (float)0.022568, (float)-0.050447, (float)0.044694, (float)-0.088052, (float)-0.022418, (float)0.046205, (float)0.016767, (float)-0.006594, (float)0.036991, (float)-0.069299, (float)-0.029071, (float)-0.060822, (float)0.003769, (float)0.098799, (float)-0.004243, (float)0.057967, (float)-0.093178, (float)-0.099343, (float)-0.100569, (float)-0.000796, (float)0.021643, (float)-0.084089, (float)-0.022391, (float)0.075777, (float)-0.032519, (float)0.081620, (float)0.055839, (float)-0.073307, (float)0.000651, (float)-0.062549, (float)0.041388, (float)-0.034778, (float)0.041743, (float)-0.090167, (float)0.076087, (float)0.040761, (float)-0.061195, (float)-0.042339, (float)-0.035109, (float)0.045258, (float)0.057178, (float)-0.026921, (float)0.059426, (float)0.030879, (float)-0.096240, (float)-0.105551, (float)0.038425, (float)0.009207, (float)-0.108471, (float)-0.073255, (float)0.032369, (float)0.076597, (float)-0.054170, (float)-0.075856, (float)-0.027868, (float)-0.082301, (float)0.056016, (float)-0.104450, (float)0.046057, (float)-0.069946, (float)0.060389, (float)0.101492, (float)0.083789, (float)0.009471, (float)0.101598, (float)0.087153, (float)0.041395, (float)0.047692, (float)-0.036329, (float)0.017234, (float)0.078113, (float)-0.108315, (float)0.024378, (float)0.090876, (float)0.018125, (float)-0.012614, (float)-0.009774, (float)-0.039836, (float)-0.036212, (float)-0.070434, (float)0.001229, (float)-0.048606, (float)-0.103865, (float)0.022066, (float)-0.081483, (float)-0.074000, (float)-0.032046, (float)-0.049501, (float)-0.035617, (float)-0.048278, (float)-0.065139, (float)0.059428, (float)0.065432, (float)-0.075881, (float)0.009768, (float)-0.101006, (float)-0.053436, (float)-0.028808, (float)0.029351, (float)-0.059764, (float)0.057315, (float)-0.080813, (float)-0.049301, (float)0.015364, (float)0.044531, (float)-0.107311, (float)0.024092, (float)-0.039331, (float)-0.103555, (float)-0.079577, (float)0.054978, (float)0.001140, (float)0.002529, (float)0.027512, (float)0.103280, (float)0.026434, (float)-0.083021, (float)0.054316, (float)-0.089931, (float)-0.016634, (float)0.025743, (float)0.031883, (float)-0.066354, (float)-0.105397, (float)-0.001965, (float)0.007055, (float)-0.044105, (float)0.039123, (float)-0.055542, (float)-0.077596, (float)0.000320, (float)-0.018303, (float)0.080768, (float)-0.015230, (float)-0.036603, (float)0.073281, (float)-0.008529, (float)0.092171, (float)0.092192, (float)0.062930, (float)0.046490, (float)0.032783, (float)0.009453, (float)0.005850, (float)-0.036920, (float)-0.057229, (float)0.052171, (float)0.023863, (float)0.057099, (float)0.079916, (float)-0.079862, (float)-0.042466, (float)0.104252, (float)-0.042889, (float)-0.032088, (float)-0.094868, (float)0.095555, (float)-0.059560, (float)0.012444, (float)0.082581, (float)0.007071, (float)0.088513, (float)-0.031910, (float)0.024311, (float)0.082048, (float)0.068908, (float)-0.032300, (float)0.053444, (float)-0.095808, (float)-0.007074, (float)0.093693, (float)-0.024909, (float)-0.043358, (float)-0.087062, (float)-0.090816, (float)0.022220, (float)0.083293, (float)0.039717, (float)-0.010215, (float)-0.019580, (float)-0.022149, (float)-0.034954, (float)0.069600, (float)-0.014078, (float)0.097535, (float)-0.018155, (float)0.078759, (float)-0.024616, (float)0.090525, (float)-0.088344, (float)0.077031, (float)0.023053, (float)-0.090771, (float)-0.063575, (float)-0.089232, (float)0.014499, (float)0.038093, (float)-0.033826, (float)-0.010511, (float)-0.012840, (float)0.063819, (float)-0.049717, (float)0.013726, (float)0.034912, (float)-0.103561, (float)-0.025387, (float)-0.023081, (float)0.001143, (float)0.050327, (float)-0.066101, (float)0.008317, (float)-0.074058, (float)0.084656, (float)-0.038123, (float)0.106940, (float)-0.000607, (float)0.023912, (float)-0.029819, (float)0.085433, (float)-0.002317, (float)0.079006, (float)0.101937, (float)0.020556, (float)-0.069768, (float)0.050517, (float)-0.023976, (float)0.003772, (float)0.071158, (float)-0.020163, (float)-0.035100, (float)0.016251, (float)0.030786, (float)-0.083116, (float)-0.085921, (float)0.093836, (float)0.099016, (float)0.061731, (float)0.027281, (float)-0.061824, (float)-0.066856, (float)0.106603, (float)-0.041826, (float)0.062703, (float)0.085510, (float)-0.000497, (float)0.058543, (float)0.003840, (float)0.079658, (float)-0.027568, (float)0.093949, (float)-0.000884, (float)0.086609, (float)0.107249, (float)-0.037643, (float)0.007986, (float)0.087096, (float)0.018908, (float)0.067048, (float)0.090737, (float)-0.040254, (float)-0.019909, (float)-0.101810, (float)0.002203, (float)-0.049097, (float)0.100996, (float)0.024320, (float)0.052374, (float)-0.084661, (float)0.059653, (float)-0.018972, (float)0.065465, (float)-0.029873, (float)0.043568, (float)0.057272, (float)0.034575, (float)0.027869, (float)0.107872, (float)0.090546, (float)0.010911, (float)0.043348, (float)-0.016302, (float)-0.008462, (float)0.092321, (float)-0.035105, (float)-0.072676, (float)0.079122, (float)0.092008, (float)-0.008498, (float)0.029312, (float)0.086154, (float)-0.028450, (float)-0.005267, (float)0.104538, (float)-0.034387, (float)0.013156, (float)0.107464, (float)0.067041, (float)0.064125, (float)-0.008582};	// L225, [2042659,2042659)
  #pragma HLS bind_storage variable=v164 type=ram_t2p impl=bram

  hls::stream<bool> v165;	// L226, [0,0)
  forward_node6(v159, v158, v136, v165, v141);	// L227, [0,1146606)
  hls::stream<bool> v166;	// L228, [1146606,1146606)
  forward_node5(v165, v142, v166, v143);	// L229, [1146606,1151904)
  hls::stream<bool> v167;	// L230, [1151904,1151904)
  forward_node4(v166, v144, v160, v138, v167, v145);	// L231, [1151904,1879977)
  float v168[16][5][5];	// L232, [1879977,1879977)
  #pragma HLS bind_storage variable=v168 type=ram_t2p impl=bram

  forward_node3(v167, v146, v168);	// L233, [1879977,1881783)
  float v169[120];	// L234, [1881783,1881783)
  #pragma HLS bind_storage variable=v169 type=ram_t2p impl=bram

  forward_node2(v168, v161, v139, v169);	// L235, [1881783,2016189)
  float v170[84];	// L236, [2016189,2016189)
  #pragma HLS bind_storage variable=v170 type=ram_t2p impl=bram

  forward_node1(v162, v140, v169, v170);	// L237, [2016189,2042659)
  forward_node0(v170, v164, v163, v137);	// L238, [2042659,2044664)
}

