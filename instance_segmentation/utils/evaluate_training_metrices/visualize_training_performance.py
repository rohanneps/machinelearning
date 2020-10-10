import pandas as pd
from matplotlib import pyplot as plt

# ENTER TRAINING STRING HERE
TRAINING_STRING ='''Epoch 1/60
100/100 [==============================] - 412s 4s/step - loss: 2.5518 - rpn_class_loss: 0.2131 - rpn_bbox_loss: 1.1578 - mrcnn_class_loss: 0.0700 - mrcnn_bbox_loss: 0.6626 - mrcnn_mask_loss: 0.4484 - val_loss: 1.9776 - val_rpn_class_loss: 0.1471 - val_rpn_bbox_loss: 0.7797 - val_mrcnn_class_loss: 0.0906 - val_mrcnn_bbox_loss: 0.5799 - val_mrcnn_mask_loss: 0.3803
Epoch 2/60
100/100 [==============================] - 71s 708ms/step - loss: 1.9652 - rpn_class_loss: 0.1350 - rpn_bbox_loss: 0.8564 - mrcnn_class_loss: 0.0619 - mrcnn_bbox_loss: 0.5408 - mrcnn_mask_loss: 0.3712 - val_loss: 1.8073 - val_rpn_class_loss: 0.1201 - val_rpn_bbox_loss: 0.7593 - val_mrcnn_class_loss: 0.1183 - val_mrcnn_bbox_loss: 0.4832 - val_mrcnn_mask_loss: 0.3264
Epoch 3/60
100/100 [==============================] - 83s 834ms/step - loss: 1.7557 - rpn_class_loss: 0.1208 - rpn_bbox_loss: 0.8432 - mrcnn_class_loss: 0.0631 - mrcnn_bbox_loss: 0.4036 - mrcnn_mask_loss: 0.3249 - val_loss: 1.8060 - val_rpn_class_loss: 0.1241 - val_rpn_bbox_loss: 0.7664 - val_mrcnn_class_loss: 0.1053 - val_mrcnn_bbox_loss: 0.4749 - val_mrcnn_mask_loss: 0.3352
Epoch 4/60
100/100 [==============================] - 82s 825ms/step - loss: 1.6675 - rpn_class_loss: 0.1116 - rpn_bbox_loss: 0.7947 - mrcnn_class_loss: 0.0677 - mrcnn_bbox_loss: 0.3668 - mrcnn_mask_loss: 0.3267 - val_loss: 1.7342 - val_rpn_class_loss: 0.1066 - val_rpn_bbox_loss: 0.7513 - val_mrcnn_class_loss: 0.1037 - val_mrcnn_bbox_loss: 0.3857 - val_mrcnn_mask_loss: 0.3869
Epoch 5/60
100/100 [==============================] - 86s 857ms/step - loss: 1.5487 - rpn_class_loss: 0.1041 - rpn_bbox_loss: 0.7617 - mrcnn_class_loss: 0.0696 - mrcnn_bbox_loss: 0.3089 - mrcnn_mask_loss: 0.3044 - val_loss: 1.7016 - val_rpn_class_loss: 0.1025 - val_rpn_bbox_loss: 0.7121 - val_mrcnn_class_loss: 0.1098 - val_mrcnn_bbox_loss: 0.4114 - val_mrcnn_mask_loss: 0.3657
Epoch 6/60
100/100 [==============================] - 115s 1s/step - loss: 1.5111 - rpn_class_loss: 0.0991 - rpn_bbox_loss: 0.7383 - mrcnn_class_loss: 0.0690 - mrcnn_bbox_loss: 0.3000 - mrcnn_mask_loss: 0.3047 - val_loss: 1.6256 - val_rpn_class_loss: 0.1131 - val_rpn_bbox_loss: 0.7158 - val_mrcnn_class_loss: 0.1157 - val_mrcnn_bbox_loss: 0.3582 - val_mrcnn_mask_loss: 0.3228
Epoch 7/60
100/100 [==============================] - 109s 1s/step - loss: 1.4980 - rpn_class_loss: 0.1098 - rpn_bbox_loss: 0.7085 - mrcnn_class_loss: 0.0733 - mrcnn_bbox_loss: 0.2964 - mrcnn_mask_loss: 0.3100 - val_loss: 1.6066 - val_rpn_class_loss: 0.0998 - val_rpn_bbox_loss: 0.7027 - val_mrcnn_class_loss: 0.1232 - val_mrcnn_bbox_loss: 0.3482 - val_mrcnn_mask_loss: 0.3326
Epoch 8/60
100/100 [==============================] - 109s 1s/step - loss: 1.4508 - rpn_class_loss: 0.0931 - rpn_bbox_loss: 0.7049 - mrcnn_class_loss: 0.0785 - mrcnn_bbox_loss: 0.2766 - mrcnn_mask_loss: 0.2977 - val_loss: 1.6157 - val_rpn_class_loss: 0.0886 - val_rpn_bbox_loss: 0.6926 - val_mrcnn_class_loss: 0.1361 - val_mrcnn_bbox_loss: 0.3547 - val_mrcnn_mask_loss: 0.3437
Epoch 9/60
100/100 [==============================] - 114s 1s/step - loss: 1.3027 - rpn_class_loss: 0.0794 - rpn_bbox_loss: 0.6099 - mrcnn_class_loss: 0.0852 - mrcnn_bbox_loss: 0.2459 - mrcnn_mask_loss: 0.2823 - val_loss: 1.6693 - val_rpn_class_loss: 0.1058 - val_rpn_bbox_loss: 0.7343 - val_mrcnn_class_loss: 0.1425 - val_mrcnn_bbox_loss: 0.3418 - val_mrcnn_mask_loss: 0.3449
Epoch 10/60
100/100 [==============================] - 101s 1s/step - loss: 1.3902 - rpn_class_loss: 0.0862 - rpn_bbox_loss: 0.6711 - mrcnn_class_loss: 0.0863 - mrcnn_bbox_loss: 0.2546 - mrcnn_mask_loss: 0.2919 - val_loss: 1.5808 - val_rpn_class_loss: 0.0965 - val_rpn_bbox_loss: 0.6676 - val_mrcnn_class_loss: 0.1136 - val_mrcnn_bbox_loss: 0.3431 - val_mrcnn_mask_loss: 0.3600
Epoch 11/60
100/100 [==============================] - 118s 1s/step - loss: 1.2174 - rpn_class_loss: 0.0779 - rpn_bbox_loss: 0.5646 - mrcnn_class_loss: 0.0897 - mrcnn_bbox_loss: 0.2196 - mrcnn_mask_loss: 0.2656 - val_loss: 1.6245 - val_rpn_class_loss: 0.1122 - val_rpn_bbox_loss: 0.7016 - val_mrcnn_class_loss: 0.1450 - val_mrcnn_bbox_loss: 0.3199 - val_mrcnn_mask_loss: 0.3458
Epoch 12/60
100/100 [==============================] - 107s 1s/step - loss: 1.1995 - rpn_class_loss: 0.0764 - rpn_bbox_loss: 0.5430 - mrcnn_class_loss: 0.0934 - mrcnn_bbox_loss: 0.2138 - mrcnn_mask_loss: 0.2729 - val_loss: 1.5835 - val_rpn_class_loss: 0.0850 - val_rpn_bbox_loss: 0.6785 - val_mrcnn_class_loss: 0.1614 - val_mrcnn_bbox_loss: 0.3162 - val_mrcnn_mask_loss: 0.3424
Epoch 13/60
100/100 [==============================] - 119s 1s/step - loss: 1.2059 - rpn_class_loss: 0.0774 - rpn_bbox_loss: 0.5509 - mrcnn_class_loss: 0.0833 - mrcnn_bbox_loss: 0.2201 - mrcnn_mask_loss: 0.2742 - val_loss: 1.6144 - val_rpn_class_loss: 0.1030 - val_rpn_bbox_loss: 0.6858 - val_mrcnn_class_loss: 0.1582 - val_mrcnn_bbox_loss: 0.3129 - val_mrcnn_mask_loss: 0.3545
Epoch 14/60
100/100 [==============================] - 105s 1s/step - loss: 1.1828 - rpn_class_loss: 0.0748 - rpn_bbox_loss: 0.5471 - mrcnn_class_loss: 0.0924 - mrcnn_bbox_loss: 0.2121 - mrcnn_mask_loss: 0.2563 - val_loss: 1.6240 - val_rpn_class_loss: 0.0985 - val_rpn_bbox_loss: 0.6863 - val_mrcnn_class_loss: 0.1580 - val_mrcnn_bbox_loss: 0.3168 - val_mrcnn_mask_loss: 0.3644
Epoch 15/60
100/100 [==============================] - 116s 1s/step - loss: 1.1640 - rpn_class_loss: 0.0757 - rpn_bbox_loss: 0.5264 - mrcnn_class_loss: 0.0921 - mrcnn_bbox_loss: 0.1983 - mrcnn_mask_loss: 0.2715 - val_loss: 1.5986 - val_rpn_class_loss: 0.1167 - val_rpn_bbox_loss: 0.6862 - val_mrcnn_class_loss: 0.1572 - val_mrcnn_bbox_loss: 0.2953 - val_mrcnn_mask_loss: 0.3431
Epoch 16/60
100/100 [==============================] - 107s 1s/step - loss: 1.0681 - rpn_class_loss: 0.0687 - rpn_bbox_loss: 0.4577 - mrcnn_class_loss: 0.0916 - mrcnn_bbox_loss: 0.1950 - mrcnn_mask_loss: 0.2551 - val_loss: 1.6050 - val_rpn_class_loss: 0.0885 - val_rpn_bbox_loss: 0.6790 - val_mrcnn_class_loss: 0.1735 - val_mrcnn_bbox_loss: 0.2952 - val_mrcnn_mask_loss: 0.3687
Epoch 17/60
100/100 [==============================] - 114s 1s/step - loss: 1.1073 - rpn_class_loss: 0.0763 - rpn_bbox_loss: 0.4772 - mrcnn_class_loss: 0.0927 - mrcnn_bbox_loss: 0.1932 - mrcnn_mask_loss: 0.2678 - val_loss: 1.6182 - val_rpn_class_loss: 0.1086 - val_rpn_bbox_loss: 0.6770 - val_mrcnn_class_loss: 0.1814 - val_mrcnn_bbox_loss: 0.3031 - val_mrcnn_mask_loss: 0.3481
Epoch 18/60
100/100 [==============================] - 116s 1s/step - loss: 1.0182 - rpn_class_loss: 0.0640 - rpn_bbox_loss: 0.4091 - mrcnn_class_loss: 0.1049 - mrcnn_bbox_loss: 0.1811 - mrcnn_mask_loss: 0.2591 - val_loss: 1.5424 - val_rpn_class_loss: 0.0860 - val_rpn_bbox_loss: 0.6277 - val_mrcnn_class_loss: 0.1774 - val_mrcnn_bbox_loss: 0.2981 - val_mrcnn_mask_loss: 0.3531
Epoch 19/60
100/100 [==============================] - 117s 1s/step - loss: 1.0322 - rpn_class_loss: 0.0667 - rpn_bbox_loss: 0.4426 - mrcnn_class_loss: 0.0868 - mrcnn_bbox_loss: 0.1785 - mrcnn_mask_loss: 0.2577 - val_loss: 1.6998 - val_rpn_class_loss: 0.1452 - val_rpn_bbox_loss: 0.7233 - val_mrcnn_class_loss: 0.1759 - val_mrcnn_bbox_loss: 0.3055 - val_mrcnn_mask_loss: 0.3497
Epoch 20/60
100/100 [==============================] - 117s 1s/step - loss: 0.9690 - rpn_class_loss: 0.0610 - rpn_bbox_loss: 0.3727 - mrcnn_class_loss: 0.0984 - mrcnn_bbox_loss: 0.1772 - mrcnn_mask_loss: 0.2598 - val_loss: 1.6464 - val_rpn_class_loss: 0.0882 - val_rpn_bbox_loss: 0.6466 - val_mrcnn_class_loss: 0.1922 - val_mrcnn_bbox_loss: 0.3192 - val_mrcnn_mask_loss: 0.4002
Epoch 21/60
100/100 [==============================] - 110s 1s/step - loss: 1.0480 - rpn_class_loss: 0.0652 - rpn_bbox_loss: 0.4294 - mrcnn_class_loss: 0.0974 - mrcnn_bbox_loss: 0.1880 - mrcnn_mask_loss: 0.2680 - val_loss: 1.5549 - val_rpn_class_loss: 0.0979 - val_rpn_bbox_loss: 0.6339 - val_mrcnn_class_loss: 0.1560 - val_mrcnn_bbox_loss: 0.3018 - val_mrcnn_mask_loss: 0.3652
Epoch 22/60
100/100 [==============================] - 115s 1s/step - loss: 0.9314 - rpn_class_loss: 0.0645 - rpn_bbox_loss: 0.3651 - mrcnn_class_loss: 0.0883 - mrcnn_bbox_loss: 0.1666 - mrcnn_mask_loss: 0.2469 - val_loss: 1.6041 - val_rpn_class_loss: 0.1094 - val_rpn_bbox_loss: 0.6505 - val_mrcnn_class_loss: 0.1572 - val_mrcnn_bbox_loss: 0.3155 - val_mrcnn_mask_loss: 0.3715
Epoch 23/60
100/100 [==============================] - 120s 1s/step - loss: 0.9492 - rpn_class_loss: 0.0584 - rpn_bbox_loss: 0.3986 - mrcnn_class_loss: 0.0965 - mrcnn_bbox_loss: 0.1588 - mrcnn_mask_loss: 0.2368 - val_loss: 1.6463 - val_rpn_class_loss: 0.1110 - val_rpn_bbox_loss: 0.6527 - val_mrcnn_class_loss: 0.1742 - val_mrcnn_bbox_loss: 0.3252 - val_mrcnn_mask_loss: 0.3832
Epoch 24/60
100/100 [==============================] - 115s 1s/step - loss: 0.9295 - rpn_class_loss: 0.0539 - rpn_bbox_loss: 0.3455 - mrcnn_class_loss: 0.1013 - mrcnn_bbox_loss: 0.1733 - mrcnn_mask_loss: 0.2556 - val_loss: 1.5713 - val_rpn_class_loss: 0.1108 - val_rpn_bbox_loss: 0.6384 - val_mrcnn_class_loss: 0.1551 - val_mrcnn_bbox_loss: 0.3083 - val_mrcnn_mask_loss: 0.3586
Epoch 25/60
100/100 [==============================] - 116s 1s/step - loss: 0.8406 - rpn_class_loss: 0.0512 - rpn_bbox_loss: 0.3149 - mrcnn_class_loss: 0.0923 - mrcnn_bbox_loss: 0.1420 - mrcnn_mask_loss: 0.2402 - val_loss: 1.5262 - val_rpn_class_loss: 0.0793 - val_rpn_bbox_loss: 0.6357 - val_mrcnn_class_loss: 0.1801 - val_mrcnn_bbox_loss: 0.2909 - val_mrcnn_mask_loss: 0.3401
Epoch 26/60
100/100 [==============================] - 122s 1s/step - loss: 0.9455 - rpn_class_loss: 0.0524 - rpn_bbox_loss: 0.3785 - mrcnn_class_loss: 0.0949 - mrcnn_bbox_loss: 0.1702 - mrcnn_mask_loss: 0.2495 - val_loss: 1.7041 - val_rpn_class_loss: 0.1191 - val_rpn_bbox_loss: 0.6704 - val_mrcnn_class_loss: 0.2218 - val_mrcnn_bbox_loss: 0.3220 - val_mrcnn_mask_loss: 0.3708
Epoch 27/60
100/100 [==============================] - 114s 1s/step - loss: 0.9036 - rpn_class_loss: 0.0491 - rpn_bbox_loss: 0.3521 - mrcnn_class_loss: 0.1027 - mrcnn_bbox_loss: 0.1543 - mrcnn_mask_loss: 0.2454 - val_loss: 1.6798 - val_rpn_class_loss: 0.1078 - val_rpn_bbox_loss: 0.6677 - val_mrcnn_class_loss: 0.1990 - val_mrcnn_bbox_loss: 0.3170 - val_mrcnn_mask_loss: 0.3884
Epoch 28/60
100/100 [==============================] - 117s 1s/step - loss: 0.8493 - rpn_class_loss: 0.0576 - rpn_bbox_loss: 0.3337 - mrcnn_class_loss: 0.0932 - mrcnn_bbox_loss: 0.1395 - mrcnn_mask_loss: 0.2253 - val_loss: 1.6231 - val_rpn_class_loss: 0.1134 - val_rpn_bbox_loss: 0.6117 - val_mrcnn_class_loss: 0.2142 - val_mrcnn_bbox_loss: 0.3036 - val_mrcnn_mask_loss: 0.3802
Epoch 29/60
100/100 [==============================] - 116s 1s/step - loss: 0.8401 - rpn_class_loss: 0.0498 - rpn_bbox_loss: 0.3300 - mrcnn_class_loss: 0.0917 - mrcnn_bbox_loss: 0.1396 - mrcnn_mask_loss: 0.2289 - val_loss: 1.7251 - val_rpn_class_loss: 0.1364 - val_rpn_bbox_loss: 0.6442 - val_mrcnn_class_loss: 0.2299 - val_mrcnn_bbox_loss: 0.3087 - val_mrcnn_mask_loss: 0.4059
Epoch 30/60
100/100 [==============================] - 115s 1s/step - loss: 0.9291 - rpn_class_loss: 0.0556 - rpn_bbox_loss: 0.3794 - mrcnn_class_loss: 0.0854 - mrcnn_bbox_loss: 0.1643 - mrcnn_mask_loss: 0.2445 - val_loss: 1.5660 - val_rpn_class_loss: 0.1193 - val_rpn_bbox_loss: 0.6289 - val_mrcnn_class_loss: 0.1596 - val_mrcnn_bbox_loss: 0.2998 - val_mrcnn_mask_loss: 0.3584
Epoch 31/60
100/100 [==============================] - 129s 1s/step - loss: 0.9477 - rpn_class_loss: 0.0486 - rpn_bbox_loss: 0.3670 - mrcnn_class_loss: 0.0961 - mrcnn_bbox_loss: 0.1745 - mrcnn_mask_loss: 0.2615 - val_loss: 1.6617 - val_rpn_class_loss: 0.1351 - val_rpn_bbox_loss: 0.6804 - val_mrcnn_class_loss: 0.1976 - val_mrcnn_bbox_loss: 0.2932 - val_mrcnn_mask_loss: 0.3554
Epoch 32/60
100/100 [==============================] - 104s 1s/step - loss: 0.8432 - rpn_class_loss: 0.0535 - rpn_bbox_loss: 0.3327 - mrcnn_class_loss: 0.0886 - mrcnn_bbox_loss: 0.1410 - mrcnn_mask_loss: 0.2274 - val_loss: 1.6387 - val_rpn_class_loss: 0.1110 - val_rpn_bbox_loss: 0.6263 - val_mrcnn_class_loss: 0.1911 - val_mrcnn_bbox_loss: 0.3086 - val_mrcnn_mask_loss: 0.4017
Epoch 33/60
100/100 [==============================] - 129s 1s/step - loss: 0.8905 - rpn_class_loss: 0.0483 - rpn_bbox_loss: 0.3435 - mrcnn_class_loss: 0.0951 - mrcnn_bbox_loss: 0.1570 - mrcnn_mask_loss: 0.2466 - val_loss: 1.6076 - val_rpn_class_loss: 0.1184 - val_rpn_bbox_loss: 0.6048 - val_mrcnn_class_loss: 0.2454 - val_mrcnn_bbox_loss: 0.2844 - val_mrcnn_mask_loss: 0.3545
Epoch 34/60
100/100 [==============================] - 114s 1s/step - loss: 0.8912 - rpn_class_loss: 0.0530 - rpn_bbox_loss: 0.3391 - mrcnn_class_loss: 0.1003 - mrcnn_bbox_loss: 0.1564 - mrcnn_mask_loss: 0.2424 - val_loss: 1.6501 - val_rpn_class_loss: 0.1072 - val_rpn_bbox_loss: 0.6367 - val_mrcnn_class_loss: 0.2091 - val_mrcnn_bbox_loss: 0.3163 - val_mrcnn_mask_loss: 0.3808
Epoch 35/60
100/100 [==============================] - 113s 1s/step - loss: 0.7505 - rpn_class_loss: 0.0384 - rpn_bbox_loss: 0.2727 - mrcnn_class_loss: 0.0879 - mrcnn_bbox_loss: 0.1294 - mrcnn_mask_loss: 0.2221 - val_loss: 1.6765 - val_rpn_class_loss: 0.1169 - val_rpn_bbox_loss: 0.6572 - val_mrcnn_class_loss: 0.2364 - val_mrcnn_bbox_loss: 0.2844 - val_mrcnn_mask_loss: 0.3815
Epoch 36/60
100/100 [==============================] - 107s 1s/step - loss: 0.8622 - rpn_class_loss: 0.0543 - rpn_bbox_loss: 0.3279 - mrcnn_class_loss: 0.0948 - mrcnn_bbox_loss: 0.1463 - mrcnn_mask_loss: 0.2388 - val_loss: 1.6281 - val_rpn_class_loss: 0.0940 - val_rpn_bbox_loss: 0.6172 - val_mrcnn_class_loss: 0.2146 - val_mrcnn_bbox_loss: 0.2997 - val_mrcnn_mask_loss: 0.4025
Epoch 37/60
100/100 [==============================] - 132s 1s/step - loss: 0.8128 - rpn_class_loss: 0.0525 - rpn_bbox_loss: 0.2931 - mrcnn_class_loss: 0.0857 - mrcnn_bbox_loss: 0.1478 - mrcnn_mask_loss: 0.2336 - val_loss: 1.7022 - val_rpn_class_loss: 0.1435 - val_rpn_bbox_loss: 0.6579 - val_mrcnn_class_loss: 0.2205 - val_mrcnn_bbox_loss: 0.2852 - val_mrcnn_mask_loss: 0.3951
Epoch 38/60
100/100 [==============================] - 110s 1s/step - loss: 0.7389 - rpn_class_loss: 0.0416 - rpn_bbox_loss: 0.2595 - mrcnn_class_loss: 0.0849 - mrcnn_bbox_loss: 0.1274 - mrcnn_mask_loss: 0.2255 - val_loss: 1.6200 - val_rpn_class_loss: 0.1150 - val_rpn_bbox_loss: 0.6371 - val_mrcnn_class_loss: 0.2203 - val_mrcnn_bbox_loss: 0.2784 - val_mrcnn_mask_loss: 0.3691
Epoch 39/60
100/100 [==============================] - 125s 1s/step - loss: 0.7822 - rpn_class_loss: 0.0461 - rpn_bbox_loss: 0.2805 - mrcnn_class_loss: 0.0865 - mrcnn_bbox_loss: 0.1386 - mrcnn_mask_loss: 0.2305 - val_loss: 1.6705 - val_rpn_class_loss: 0.1221 - val_rpn_bbox_loss: 0.6644 - val_mrcnn_class_loss: 0.1964 - val_mrcnn_bbox_loss: 0.2904 - val_mrcnn_mask_loss: 0.3973
Epoch 40/60
100/100 [==============================] - 116s 1s/step - loss: 0.7734 - rpn_class_loss: 0.0511 - rpn_bbox_loss: 0.2749 - mrcnn_class_loss: 0.0908 - mrcnn_bbox_loss: 0.1327 - mrcnn_mask_loss: 0.2238 - val_loss: 1.6649 - val_rpn_class_loss: 0.1072 - val_rpn_bbox_loss: 0.6342 - val_mrcnn_class_loss: 0.2167 - val_mrcnn_bbox_loss: 0.3129 - val_mrcnn_mask_loss: 0.3939
Epoch 41/60
100/100 [==============================] - 119s 1s/step - loss: 0.8195 - rpn_class_loss: 0.0488 - rpn_bbox_loss: 0.3044 - mrcnn_class_loss: 0.0945 - mrcnn_bbox_loss: 0.1386 - mrcnn_mask_loss: 0.2332 - val_loss: 1.6636 - val_rpn_class_loss: 0.1311 - val_rpn_bbox_loss: 0.6197 - val_mrcnn_class_loss: 0.2583 - val_mrcnn_bbox_loss: 0.2822 - val_mrcnn_mask_loss: 0.3722
Epoch 42/60
100/100 [==============================] - 116s 1s/step - loss: 0.7652 - rpn_class_loss: 0.0422 - rpn_bbox_loss: 0.2854 - mrcnn_class_loss: 0.0810 - mrcnn_bbox_loss: 0.1331 - mrcnn_mask_loss: 0.2235 - val_loss: 1.7161 - val_rpn_class_loss: 0.1136 - val_rpn_bbox_loss: 0.6556 - val_mrcnn_class_loss: 0.2576 - val_mrcnn_bbox_loss: 0.2941 - val_mrcnn_mask_loss: 0.3952
Epoch 43/60
100/100 [==============================] - 120s 1s/step - loss: 0.8042 - rpn_class_loss: 0.0428 - rpn_bbox_loss: 0.2967 - mrcnn_class_loss: 0.0930 - mrcnn_bbox_loss: 0.1414 - mrcnn_mask_loss: 0.2304 - val_loss: 1.6665 - val_rpn_class_loss: 0.1377 - val_rpn_bbox_loss: 0.6611 - val_mrcnn_class_loss: 0.2299 - val_mrcnn_bbox_loss: 0.2914 - val_mrcnn_mask_loss: 0.3464
Epoch 44/60
100/100 [==============================] - 123s 1s/step - loss: 0.8165 - rpn_class_loss: 0.0434 - rpn_bbox_loss: 0.2894 - mrcnn_class_loss: 0.0980 - mrcnn_bbox_loss: 0.1470 - mrcnn_mask_loss: 0.2386 - val_loss: 1.6714 - val_rpn_class_loss: 0.1262 - val_rpn_bbox_loss: 0.6369 - val_mrcnn_class_loss: 0.2048 - val_mrcnn_bbox_loss: 0.3030 - val_mrcnn_mask_loss: 0.4006
Epoch 45/60
100/100 [==============================] - 106s 1s/step - loss: 0.7727 - rpn_class_loss: 0.0460 - rpn_bbox_loss: 0.3002 - mrcnn_class_loss: 0.0829 - mrcnn_bbox_loss: 0.1292 - mrcnn_mask_loss: 0.2143 - val_loss: 1.5944 - val_rpn_class_loss: 0.1141 - val_rpn_bbox_loss: 0.6376 - val_mrcnn_class_loss: 0.2310 - val_mrcnn_bbox_loss: 0.2672 - val_mrcnn_mask_loss: 0.3444
Epoch 46/60
100/100 [==============================] - 129s 1s/step - loss: 0.8139 - rpn_class_loss: 0.0421 - rpn_bbox_loss: 0.3294 - mrcnn_class_loss: 0.0848 - mrcnn_bbox_loss: 0.1333 - mrcnn_mask_loss: 0.2243 - val_loss: 1.7166 - val_rpn_class_loss: 0.1547 - val_rpn_bbox_loss: 0.6449 - val_mrcnn_class_loss: 0.2453 - val_mrcnn_bbox_loss: 0.3041 - val_mrcnn_mask_loss: 0.3676
Epoch 47/60
100/100 [==============================] - 103s 1s/step - loss: 0.7536 - rpn_class_loss: 0.0392 - rpn_bbox_loss: 0.2761 - mrcnn_class_loss: 0.0752 - mrcnn_bbox_loss: 0.1370 - mrcnn_mask_loss: 0.2261 - val_loss: 1.7755 - val_rpn_class_loss: 0.1109 - val_rpn_bbox_loss: 0.6838 - val_mrcnn_class_loss: 0.2735 - val_mrcnn_bbox_loss: 0.3042 - val_mrcnn_mask_loss: 0.4031
Epoch 48/60
100/100 [==============================] - 117s 1s/step - loss: 0.7670 - rpn_class_loss: 0.0439 - rpn_bbox_loss: 0.2519 - mrcnn_class_loss: 0.1005 - mrcnn_bbox_loss: 0.1423 - mrcnn_mask_loss: 0.2284 - val_loss: 1.7861 - val_rpn_class_loss: 0.1318 - val_rpn_bbox_loss: 0.6678 - val_mrcnn_class_loss: 0.3312 - val_mrcnn_bbox_loss: 0.3021 - val_mrcnn_mask_loss: 0.3531
Epoch 49/60
100/100 [==============================] - 120s 1s/step - loss: 0.7924 - rpn_class_loss: 0.0505 - rpn_bbox_loss: 0.3153 - mrcnn_class_loss: 0.0785 - mrcnn_bbox_loss: 0.1314 - mrcnn_mask_loss: 0.2167 - val_loss: 1.6756 - val_rpn_class_loss: 0.1422 - val_rpn_bbox_loss: 0.6422 - val_mrcnn_class_loss: 0.2630 - val_mrcnn_bbox_loss: 0.2745 - val_mrcnn_mask_loss: 0.3537
Epoch 50/60
100/100 [==============================] - 112s 1s/step - loss: 0.7706 - rpn_class_loss: 0.0455 - rpn_bbox_loss: 0.2907 - mrcnn_class_loss: 0.0844 - mrcnn_bbox_loss: 0.1286 - mrcnn_mask_loss: 0.2215 - val_loss: 1.7463 - val_rpn_class_loss: 0.1292 - val_rpn_bbox_loss: 0.6670 - val_mrcnn_class_loss: 0.2501 - val_mrcnn_bbox_loss: 0.3194 - val_mrcnn_mask_loss: 0.3804
Epoch 51/60
100/100 [==============================] - 121s 1s/step - loss: 0.7140 - rpn_class_loss: 0.0377 - rpn_bbox_loss: 0.2630 - mrcnn_class_loss: 0.0790 - mrcnn_bbox_loss: 0.1216 - mrcnn_mask_loss: 0.2126 - val_loss: 1.7644 - val_rpn_class_loss: 0.1412 - val_rpn_bbox_loss: 0.7256 - val_mrcnn_class_loss: 0.2561 - val_mrcnn_bbox_loss: 0.2770 - val_mrcnn_mask_loss: 0.3645
Epoch 52/60
100/100 [==============================] - 114s 1s/step - loss: 0.6647 - rpn_class_loss: 0.0359 - rpn_bbox_loss: 0.2224 - mrcnn_class_loss: 0.0830 - mrcnn_bbox_loss: 0.1151 - mrcnn_mask_loss: 0.2082 - val_loss: 1.8042 - val_rpn_class_loss: 0.1172 - val_rpn_bbox_loss: 0.6388 - val_mrcnn_class_loss: 0.3484 - val_mrcnn_bbox_loss: 0.2792 - val_mrcnn_mask_loss: 0.4206
Epoch 53/60
100/100 [==============================] - 116s 1s/step - loss: 0.7281 - rpn_class_loss: 0.0408 - rpn_bbox_loss: 0.2487 - mrcnn_class_loss: 0.0928 - mrcnn_bbox_loss: 0.1274 - mrcnn_mask_loss: 0.2185 - val_loss: 1.7843 - val_rpn_class_loss: 0.1571 - val_rpn_bbox_loss: 0.6720 - val_mrcnn_class_loss: 0.2810 - val_mrcnn_bbox_loss: 0.2861 - val_mrcnn_mask_loss: 0.3882
Epoch 54/60
100/100 [==============================] - 118s 1s/step - loss: 0.7668 - rpn_class_loss: 0.0433 - rpn_bbox_loss: 0.2729 - mrcnn_class_loss: 0.0887 - mrcnn_bbox_loss: 0.1374 - mrcnn_mask_loss: 0.2244 - val_loss: 1.6793 - val_rpn_class_loss: 0.1177 - val_rpn_bbox_loss: 0.6603 - val_mrcnn_class_loss: 0.2257 - val_mrcnn_bbox_loss: 0.3009 - val_mrcnn_mask_loss: 0.3747
Epoch 55/60
100/100 [==============================] - 121s 1s/step - loss: 0.7419 - rpn_class_loss: 0.0403 - rpn_bbox_loss: 0.2623 - mrcnn_class_loss: 0.0844 - mrcnn_bbox_loss: 0.1303 - mrcnn_mask_loss: 0.2244 - val_loss: 1.7485 - val_rpn_class_loss: 0.1550 - val_rpn_bbox_loss: 0.6619 - val_mrcnn_class_loss: 0.2783 - val_mrcnn_bbox_loss: 0.2782 - val_mrcnn_mask_loss: 0.3752
Epoch 56/60
 61/100 [=================>............] - ETA: 25s - loss: 0.5930 - rpn_class_loss: 0.0332 - rpn_bbox_loss: 0.1894 - mrcnn_class_loss: 0.0756 - mrcnn_bbox_loss: 0.0969 - mrcnn_mask_loss: 0.1980'''


def generate_metric_graph(df, metric_name ='loss'):
	df['epoch'] = df['epoch'].astype(int)
	df[metric_name] = df[metric_name].astype(float)
	df['val_{}'.format(metric_name)] = df['val_{}'.format(metric_name)].astype(float)
	plt.figure(figsize=(20,10))
	plt.title('Mask Rcnn Train vs Test {} per Epoch'.format(metric_name))
	plt.plot(df['epoch'], df[metric_name], label='Train')
	plt.plot(df['epoch'], df['val_{}'.format(metric_name)], label='Test')
	best_val_metric_loss_value = df['val_{}'.format(metric_name)].min()
	best_val_metric_loss_epoch = (df['val_{}'.format(metric_name)].argmin())+1
	plt.plot((best_val_metric_loss_epoch, best_val_metric_loss_epoch), (0, best_val_metric_loss_value), ls='dotted', color='r')
	plt.xlabel('Epochs. Best epoch model is: {} with value: {}'.format(best_val_metric_loss_epoch, best_val_metric_loss_value))
	plt.ylabel(metric_name)
	plt.legend()
	# plt.show()
	plt.savefig('{}_plot.png'.format(metric_name))

if __name__ == '__main__':
	lines = TRAINING_STRING.split('\n')
	lines = list(filter(lambda x: x !='', lines))
	eval_df = pd.DataFrame(columns=['epoch'])
	epoch_row = {}
	for line in lines:
		if 'Epoch' in line:
			epoch_row['epoch'] = line.split()[-1].split('/')[0]
		else:
			metrices = line.split(' - ')
			epoch_row['epoch_time'] = metrices[1]
			for metric in metrices[2:]:
				metric_name = metric.split(':')[0].strip()
				metric_value = metric.split(':')[1].strip()
				epoch_row[metric_name] = metric_value
			eval_df = eval_df.append(epoch_row, ignore_index=True)
			epoch_row = {}

	# eval_df[['epoch','val_loss']].sort_values(by=['val_loss'], ascending=True).to_csv('Epoch_vs_loss.tsv', index=False, sep='\t')
	generate_metric_graph(eval_df, 'loss')