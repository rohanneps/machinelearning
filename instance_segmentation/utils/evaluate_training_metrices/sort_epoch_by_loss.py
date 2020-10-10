import pandas as pd

# ENTER TRAINING STRING HERE
TRAINING_STRING ='''Epoch 1/60
100/100 [==============================] - 261s 3s/step - loss: 2.2506 - rpn_class_loss: 0.0922 - rpn_bbox_loss: 1.0470 - mrcnn_class_loss: 0.1094 - mrcnn_bbox_loss: 0.6633 - mrcnn_mask_loss: 0.3387 - val_loss: 1.9880 - val_rpn_class_loss: 0.0658 - val_rpn_bbox_loss: 0.7721 - val_mrcnn_class_loss: 0.1724 - val_mrcnn_bbox_loss: 0.6159 - val_mrcnn_mask_loss: 0.3618
Epoch 2/60
100/100 [==============================] - 81s 814ms/step - loss: 1.7840 - rpn_class_loss: 0.0660 - rpn_bbox_loss: 0.8389 - mrcnn_class_loss: 0.0939 - mrcnn_bbox_loss: 0.4643 - mrcnn_mask_loss: 0.3209 - val_loss: 1.9107 - val_rpn_class_loss: 0.0638 - val_rpn_bbox_loss: 0.7841 - val_mrcnn_class_loss: 0.1475 - val_mrcnn_bbox_loss: 0.5395 - val_mrcnn_mask_loss: 0.3758
Epoch 3/60
100/100 [==============================] - 81s 810ms/step - loss: 1.4271 - rpn_class_loss: 0.0582 - rpn_bbox_loss: 0.7043 - mrcnn_class_loss: 0.1068 - mrcnn_bbox_loss: 0.2697 - mrcnn_mask_loss: 0.2881 - val_loss: 1.7780 - val_rpn_class_loss: 0.0503 - val_rpn_bbox_loss: 0.7202 - val_mrcnn_class_loss: 0.1788 - val_mrcnn_bbox_loss: 0.4553 - val_mrcnn_mask_loss: 0.3734
Epoch 4/60
100/100 [==============================] - 99s 989ms/step - loss: 1.3036 - rpn_class_loss: 0.0560 - rpn_bbox_loss: 0.6306 - mrcnn_class_loss: 0.1076 - mrcnn_bbox_loss: 0.2228 - mrcnn_mask_loss: 0.2866 - val_loss: 1.8009 - val_rpn_class_loss: 0.0570 - val_rpn_bbox_loss: 0.7475 - val_mrcnn_class_loss: 0.1555 - val_mrcnn_bbox_loss: 0.4696 - val_mrcnn_mask_loss: 0.3714
Epoch 5/60
100/100 [==============================] - 114s 1s/step - loss: 1.2428 - rpn_class_loss: 0.0514 - rpn_bbox_loss: 0.5869 - mrcnn_class_loss: 0.1122 - mrcnn_bbox_loss: 0.2175 - mrcnn_mask_loss: 0.2747 - val_loss: 1.8548 - val_rpn_class_loss: 0.0452 - val_rpn_bbox_loss: 0.7493 - val_mrcnn_class_loss: 0.1938 - val_mrcnn_bbox_loss: 0.4745 - val_mrcnn_mask_loss: 0.3921
Epoch 6/60
100/100 [==============================] - 113s 1s/step - loss: 1.0431 - rpn_class_loss: 0.0455 - rpn_bbox_loss: 0.4592 - mrcnn_class_loss: 0.1029 - mrcnn_bbox_loss: 0.1701 - mrcnn_mask_loss: 0.2655 - val_loss: 1.7144 - val_rpn_class_loss: 0.0515 - val_rpn_bbox_loss: 0.7090 - val_mrcnn_class_loss: 0.1950 - val_mrcnn_bbox_loss: 0.4110 - val_mrcnn_mask_loss: 0.3480
Epoch 7/60
100/100 [==============================] - 112s 1s/step - loss: 0.9700 - rpn_class_loss: 0.0462 - rpn_bbox_loss: 0.4144 - mrcnn_class_loss: 0.0934 - mrcnn_bbox_loss: 0.1603 - mrcnn_mask_loss: 0.2557 - val_loss: 1.7658 - val_rpn_class_loss: 0.0477 - val_rpn_bbox_loss: 0.7075 - val_mrcnn_class_loss: 0.2032 - val_mrcnn_bbox_loss: 0.4120 - val_mrcnn_mask_loss: 0.3954
Epoch 8/60
100/100 [==============================] - 116s 1s/step - loss: 0.9078 - rpn_class_loss: 0.0422 - rpn_bbox_loss: 0.3527 - mrcnn_class_loss: 0.1084 - mrcnn_bbox_loss: 0.1500 - mrcnn_mask_loss: 0.2544 - val_loss: 1.8296 - val_rpn_class_loss: 0.0512 - val_rpn_bbox_loss: 0.7317 - val_mrcnn_class_loss: 0.2435 - val_mrcnn_bbox_loss: 0.3961 - val_mrcnn_mask_loss: 0.4071
Epoch 9/60
100/100 [==============================] - 116s 1s/step - loss: 0.8079 - rpn_class_loss: 0.0391 - rpn_bbox_loss: 0.3086 - mrcnn_class_loss: 0.1007 - mrcnn_bbox_loss: 0.1190 - mrcnn_mask_loss: 0.2404 - val_loss: 1.8149 - val_rpn_class_loss: 0.0499 - val_rpn_bbox_loss: 0.7126 - val_mrcnn_class_loss: 0.2717 - val_mrcnn_bbox_loss: 0.3662 - val_mrcnn_mask_loss: 0.4145
Epoch 10/60
100/100 [==============================] - 108s 1s/step - loss: 0.7417 - rpn_class_loss: 0.0389 - rpn_bbox_loss: 0.2627 - mrcnn_class_loss: 0.0900 - mrcnn_bbox_loss: 0.1170 - mrcnn_mask_loss: 0.2331 - val_loss: 1.7814 - val_rpn_class_loss: 0.0447 - val_rpn_bbox_loss: 0.7239 - val_mrcnn_class_loss: 0.2446 - val_mrcnn_bbox_loss: 0.3571 - val_mrcnn_mask_loss: 0.4112
Epoch 11/60
100/100 [==============================] - 119s 1s/step - loss: 0.7201 - rpn_class_loss: 0.0362 - rpn_bbox_loss: 0.2514 - mrcnn_class_loss: 0.0895 - mrcnn_bbox_loss: 0.1204 - mrcnn_mask_loss: 0.2226 - val_loss: 1.8946 - val_rpn_class_loss: 0.0502 - val_rpn_bbox_loss: 0.7478 - val_mrcnn_class_loss: 0.2945 - val_mrcnn_bbox_loss: 0.3806 - val_mrcnn_mask_loss: 0.4216
Epoch 12/60
100/100 [==============================] - 118s 1s/step - loss: 0.6690 - rpn_class_loss: 0.0350 - rpn_bbox_loss: 0.2115 - mrcnn_class_loss: 0.0807 - mrcnn_bbox_loss: 0.1188 - mrcnn_mask_loss: 0.2229 - val_loss: 1.8556 - val_rpn_class_loss: 0.0458 - val_rpn_bbox_loss: 0.7270 - val_mrcnn_class_loss: 0.2668 - val_mrcnn_bbox_loss: 0.3881 - val_mrcnn_mask_loss: 0.4279
Epoch 13/60
100/100 [==============================] - 112s 1s/step - loss: 0.6445 - rpn_class_loss: 0.0338 - rpn_bbox_loss: 0.2166 - mrcnn_class_loss: 0.0722 - mrcnn_bbox_loss: 0.1102 - mrcnn_mask_loss: 0.2117 - val_loss: 1.9738 - val_rpn_class_loss: 0.0424 - val_rpn_bbox_loss: 0.7269 - val_mrcnn_class_loss: 0.3357 - val_mrcnn_bbox_loss: 0.4081 - val_mrcnn_mask_loss: 0.4608
Epoch 14/60
100/100 [==============================] - 118s 1s/step - loss: 0.6323 - rpn_class_loss: 0.0337 - rpn_bbox_loss: 0.1828 - mrcnn_class_loss: 0.0837 - mrcnn_bbox_loss: 0.1121 - mrcnn_mask_loss: 0.2200 - val_loss: 1.8821 - val_rpn_class_loss: 0.0468 - val_rpn_bbox_loss: 0.7118 - val_mrcnn_class_loss: 0.3004 - val_mrcnn_bbox_loss: 0.3663 - val_mrcnn_mask_loss: 0.4568
Epoch 15/60
100/100 [==============================] - 127s 1s/step - loss: 0.5559 - rpn_class_loss: 0.0301 - rpn_bbox_loss: 0.1558 - mrcnn_class_loss: 0.0763 - mrcnn_bbox_loss: 0.0931 - mrcnn_mask_loss: 0.2006 - val_loss: 2.0608 - val_rpn_class_loss: 0.0533 - val_rpn_bbox_loss: 0.7801 - val_mrcnn_class_loss: 0.3397 - val_mrcnn_bbox_loss: 0.3978 - val_mrcnn_mask_loss: 0.4899
Epoch 16/60
100/100 [==============================] - 103s 1s/step - loss: 0.5781 - rpn_class_loss: 0.0312 - rpn_bbox_loss: 0.1588 - mrcnn_class_loss: 0.0676 - mrcnn_bbox_loss: 0.1010 - mrcnn_mask_loss: 0.2194 - val_loss: 1.9747 - val_rpn_class_loss: 0.0380 - val_rpn_bbox_loss: 0.7054 - val_mrcnn_class_loss: 0.3779 - val_mrcnn_bbox_loss: 0.3968 - val_mrcnn_mask_loss: 0.4566
Epoch 17/60
100/100 [==============================] - 117s 1s/step - loss: 0.5630 - rpn_class_loss: 0.0286 - rpn_bbox_loss: 0.1495 - mrcnn_class_loss: 0.0657 - mrcnn_bbox_loss: 0.1077 - mrcnn_mask_loss: 0.2115 - val_loss: 1.9907 - val_rpn_class_loss: 0.0468 - val_rpn_bbox_loss: 0.7364 - val_mrcnn_class_loss: 0.3538 - val_mrcnn_bbox_loss: 0.3685 - val_mrcnn_mask_loss: 0.4851
Epoch 18/60
100/100 [==============================] - 116s 1s/step - loss: 0.4967 - rpn_class_loss: 0.0276 - rpn_bbox_loss: 0.1292 - mrcnn_class_loss: 0.0580 - mrcnn_bbox_loss: 0.0837 - mrcnn_mask_loss: 0.1981 - val_loss: 2.2302 - val_rpn_class_loss: 0.0445 - val_rpn_bbox_loss: 0.7403 - val_mrcnn_class_loss: 0.5149 - val_mrcnn_bbox_loss: 0.4187 - val_mrcnn_mask_loss: 0.5117
Epoch 19/60
100/100 [==============================] - 115s 1s/step - loss: 0.5062 - rpn_class_loss: 0.0276 - rpn_bbox_loss: 0.1316 - mrcnn_class_loss: 0.0611 - mrcnn_bbox_loss: 0.0859 - mrcnn_mask_loss: 0.2000 - val_loss: 2.0420 - val_rpn_class_loss: 0.0405 - val_rpn_bbox_loss: 0.7301 - val_mrcnn_class_loss: 0.3776 - val_mrcnn_bbox_loss: 0.3702 - val_mrcnn_mask_loss: 0.5237
Epoch 20/60
100/100 [==============================] - 122s 1s/step - loss: 0.5091 - rpn_class_loss: 0.0264 - rpn_bbox_loss: 0.1150 - mrcnn_class_loss: 0.0639 - mrcnn_bbox_loss: 0.1006 - mrcnn_mask_loss: 0.2031 - val_loss: 1.9330 - val_rpn_class_loss: 0.0518 - val_rpn_bbox_loss: 0.7772 - val_mrcnn_class_loss: 0.2815 - val_mrcnn_bbox_loss: 0.3614 - val_mrcnn_mask_loss: 0.4612
Epoch 21/60
100/100 [==============================] - 116s 1s/step - loss: 0.4783 - rpn_class_loss: 0.0249 - rpn_bbox_loss: 0.1106 - mrcnn_class_loss: 0.0627 - mrcnn_bbox_loss: 0.0856 - mrcnn_mask_loss: 0.1945 - val_loss: 2.2285 - val_rpn_class_loss: 0.0454 - val_rpn_bbox_loss: 0.7290 - val_mrcnn_class_loss: 0.5140 - val_mrcnn_bbox_loss: 0.3942 - val_mrcnn_mask_loss: 0.5460
Epoch 22/60
100/100 [==============================] - 120s 1s/step - loss: 0.4774 - rpn_class_loss: 0.0248 - rpn_bbox_loss: 0.1115 - mrcnn_class_loss: 0.0656 - mrcnn_bbox_loss: 0.0839 - mrcnn_mask_loss: 0.1915 - val_loss: 2.1424 - val_rpn_class_loss: 0.0466 - val_rpn_bbox_loss: 0.7119 - val_mrcnn_class_loss: 0.4676 - val_mrcnn_bbox_loss: 0.3803 - val_mrcnn_mask_loss: 0.5360
Epoch 23/60
100/100 [==============================] - 108s 1s/step - loss: 0.4646 - rpn_class_loss: 0.0243 - rpn_bbox_loss: 0.0968 - mrcnn_class_loss: 0.0605 - mrcnn_bbox_loss: 0.0827 - mrcnn_mask_loss: 0.2004 - val_loss: 2.2218 - val_rpn_class_loss: 0.0441 - val_rpn_bbox_loss: 0.7613 - val_mrcnn_class_loss: 0.5009 - val_mrcnn_bbox_loss: 0.3992 - val_mrcnn_mask_loss: 0.5163
Epoch 24/60
100/100 [==============================] - 128s 1s/step - loss: 0.4575 - rpn_class_loss: 0.0231 - rpn_bbox_loss: 0.1008 - mrcnn_class_loss: 0.0600 - mrcnn_bbox_loss: 0.0804 - mrcnn_mask_loss: 0.1933 - val_loss: 2.1235 - val_rpn_class_loss: 0.0527 - val_rpn_bbox_loss: 0.7635 - val_mrcnn_class_loss: 0.4287 - val_mrcnn_bbox_loss: 0.3653 - val_mrcnn_mask_loss: 0.5134
Epoch 25/60
100/100 [==============================] - 105s 1s/step - loss: 0.4533 - rpn_class_loss: 0.0227 - rpn_bbox_loss: 0.0945 - mrcnn_class_loss: 0.0592 - mrcnn_bbox_loss: 0.0823 - mrcnn_mask_loss: 0.1946 - val_loss: 2.0936 - val_rpn_class_loss: 0.0392 - val_rpn_bbox_loss: 0.7590 - val_mrcnn_class_loss: 0.3831 - val_mrcnn_bbox_loss: 0.3826 - val_mrcnn_mask_loss: 0.5296
Epoch 26/60
100/100 [==============================] - 116s 1s/step - loss: 0.4123 - rpn_class_loss: 0.0214 - rpn_bbox_loss: 0.0806 - mrcnn_class_loss: 0.0516 - mrcnn_bbox_loss: 0.0744 - mrcnn_mask_loss: 0.1843 - val_loss: 2.2045 - val_rpn_class_loss: 0.0489 - val_rpn_bbox_loss: 0.7866 - val_mrcnn_class_loss: 0.4398 - val_mrcnn_bbox_loss: 0.3638 - val_mrcnn_mask_loss: 0.5655
Epoch 27/60
100/100 [==============================] - 116s 1s/step - loss: 0.4175 - rpn_class_loss: 0.0223 - rpn_bbox_loss: 0.0821 - mrcnn_class_loss: 0.0589 - mrcnn_bbox_loss: 0.0690 - mrcnn_mask_loss: 0.1851 - val_loss: 2.3598 - val_rpn_class_loss: 0.0474 - val_rpn_bbox_loss: 0.7369 - val_mrcnn_class_loss: 0.6075 - val_mrcnn_bbox_loss: 0.3763 - val_mrcnn_mask_loss: 0.5917
Epoch 28/60
100/100 [==============================] - 121s 1s/step - loss: 0.4022 - rpn_class_loss: 0.0198 - rpn_bbox_loss: 0.0878 - mrcnn_class_loss: 0.0521 - mrcnn_bbox_loss: 0.0669 - mrcnn_mask_loss: 0.1755 - val_loss: 2.3825 - val_rpn_class_loss: 0.0458 - val_rpn_bbox_loss: 0.8387 - val_mrcnn_class_loss: 0.5162 - val_mrcnn_bbox_loss: 0.3929 - val_mrcnn_mask_loss: 0.5889
Epoch 29/60
100/100 [==============================] - 117s 1s/step - loss: 0.4135 - rpn_class_loss: 0.0199 - rpn_bbox_loss: 0.0709 - mrcnn_class_loss: 0.0569 - mrcnn_bbox_loss: 0.0764 - mrcnn_mask_loss: 0.1893 - val_loss: 2.1874 - val_rpn_class_loss: 0.0460 - val_rpn_bbox_loss: 0.8027 - val_mrcnn_class_loss: 0.4792 - val_mrcnn_bbox_loss: 0.3550 - val_mrcnn_mask_loss: 0.5045
Epoch 30/60
100/100 [==============================] - 108s 1s/step - loss: 0.3776 - rpn_class_loss: 0.0181 - rpn_bbox_loss: 0.0694 - mrcnn_class_loss: 0.0474 - mrcnn_bbox_loss: 0.0683 - mrcnn_mask_loss: 0.1745 - val_loss: 2.3894 - val_rpn_class_loss: 0.0453 - val_rpn_bbox_loss: 0.7305 - val_mrcnn_class_loss: 0.5911 - val_mrcnn_bbox_loss: 0.3955 - val_mrcnn_mask_loss: 0.6270
Epoch 31/60
100/100 [==============================] - 117s 1s/step - loss: 0.3859 - rpn_class_loss: 0.0191 - rpn_bbox_loss: 0.0696 - mrcnn_class_loss: 0.0440 - mrcnn_bbox_loss: 0.0737 - mrcnn_mask_loss: 0.1794 - val_loss: 2.3353 - val_rpn_class_loss: 0.0483 - val_rpn_bbox_loss: 0.7554 - val_mrcnn_class_loss: 0.5684 - val_mrcnn_bbox_loss: 0.3604 - val_mrcnn_mask_loss: 0.6029
Epoch 32/60
100/100 [==============================] - 120s 1s/step - loss: 0.3622 - rpn_class_loss: 0.0172 - rpn_bbox_loss: 0.0583 - mrcnn_class_loss: 0.0519 - mrcnn_bbox_loss: 0.0633 - mrcnn_mask_loss: 0.1714 - val_loss: 2.3082 - val_rpn_class_loss: 0.0531 - val_rpn_bbox_loss: 0.7464 - val_mrcnn_class_loss: 0.6061 - val_mrcnn_bbox_loss: 0.3693 - val_mrcnn_mask_loss: 0.5333
Epoch 33/60
100/100 [==============================] - 110s 1s/step - loss: 0.3807 - rpn_class_loss: 0.0180 - rpn_bbox_loss: 0.0693 - mrcnn_class_loss: 0.0462 - mrcnn_bbox_loss: 0.0667 - mrcnn_mask_loss: 0.1804 - val_loss: 2.6666 - val_rpn_class_loss: 0.0454 - val_rpn_bbox_loss: 0.7544 - val_mrcnn_class_loss: 0.7964 - val_mrcnn_bbox_loss: 0.4173 - val_mrcnn_mask_loss: 0.6531
Epoch 34/60
100/100 [==============================] - 120s 1s/step - loss: 0.3877 - rpn_class_loss: 0.0159 - rpn_bbox_loss: 0.0697 - mrcnn_class_loss: 0.0546 - mrcnn_bbox_loss: 0.0697 - mrcnn_mask_loss: 0.1778 - val_loss: 2.1841 - val_rpn_class_loss: 0.0515 - val_rpn_bbox_loss: 0.7566 - val_mrcnn_class_loss: 0.4223 - val_mrcnn_bbox_loss: 0.3772 - val_mrcnn_mask_loss: 0.5765
Epoch 35/60
100/100 [==============================] - 117s 1s/step - loss: 0.3588 - rpn_class_loss: 0.0159 - rpn_bbox_loss: 0.0593 - mrcnn_class_loss: 0.0516 - mrcnn_bbox_loss: 0.0628 - mrcnn_mask_loss: 0.1691 - val_loss: 2.4167 - val_rpn_class_loss: 0.0458 - val_rpn_bbox_loss: 0.7320 - val_mrcnn_class_loss: 0.5833 - val_mrcnn_bbox_loss: 0.4163 - val_mrcnn_mask_loss: 0.6394
Epoch 36/60
100/100 [==============================] - 107s 1s/step - loss: 0.3522 - rpn_class_loss: 0.0155 - rpn_bbox_loss: 0.0576 - mrcnn_class_loss: 0.0408 - mrcnn_bbox_loss: 0.0686 - mrcnn_mask_loss: 0.1697 - val_loss: 2.3008 - val_rpn_class_loss: 0.0491 - val_rpn_bbox_loss: 0.7682 - val_mrcnn_class_loss: 0.4953 - val_mrcnn_bbox_loss: 0.3892 - val_mrcnn_mask_loss: 0.5990
Epoch 37/60
100/100 [==============================] - 121s 1s/step - loss: 0.3672 - rpn_class_loss: 0.0153 - rpn_bbox_loss: 0.0564 - mrcnn_class_loss: 0.0540 - mrcnn_bbox_loss: 0.0698 - mrcnn_mask_loss: 0.1717 - val_loss: 2.3076 - val_rpn_class_loss: 0.0473 - val_rpn_bbox_loss: 0.7774 - val_mrcnn_class_loss: 0.5504 - val_mrcnn_bbox_loss: 0.3577 - val_mrcnn_mask_loss: 0.5748
Epoch 38/60
100/100 [==============================] - 113s 1s/step - loss: 0.3488 - rpn_class_loss: 0.0142 - rpn_bbox_loss: 0.0577 - mrcnn_class_loss: 0.0390 - mrcnn_bbox_loss: 0.0635 - mrcnn_mask_loss: 0.1744 - val_loss: 2.3679 - val_rpn_class_loss: 0.0511 - val_rpn_bbox_loss: 0.8293 - val_mrcnn_class_loss: 0.5246 - val_mrcnn_bbox_loss: 0.3622 - val_mrcnn_mask_loss: 0.6007
Epoch 39/60
100/100 [==============================] - 113s 1s/step - loss: 0.3445 - rpn_class_loss: 0.0144 - rpn_bbox_loss: 0.0558 - mrcnn_class_loss: 0.0408 - mrcnn_bbox_loss: 0.0649 - mrcnn_mask_loss: 0.1686 - val_loss: 2.5119 - val_rpn_class_loss: 0.0528 - val_rpn_bbox_loss: 0.8040 - val_mrcnn_class_loss: 0.6727 - val_mrcnn_bbox_loss: 0.3785 - val_mrcnn_mask_loss: 0.6038
Epoch 40/60
100/100 [==============================] - 124s 1s/step - loss: 0.3450 - rpn_class_loss: 0.0134 - rpn_bbox_loss: 0.0520 - mrcnn_class_loss: 0.0530 - mrcnn_bbox_loss: 0.0604 - mrcnn_mask_loss: 0.1662 - val_loss: 2.5102 - val_rpn_class_loss: 0.0566 - val_rpn_bbox_loss: 0.7748 - val_mrcnn_class_loss: 0.7521 - val_mrcnn_bbox_loss: 0.3390 - val_mrcnn_mask_loss: 0.5877
Epoch 41/60
100/100 [==============================] - 112s 1s/step - loss: 0.3398 - rpn_class_loss: 0.0133 - rpn_bbox_loss: 0.0504 - mrcnn_class_loss: 0.0432 - mrcnn_bbox_loss: 0.0649 - mrcnn_mask_loss: 0.1679 - val_loss: 2.5156 - val_rpn_class_loss: 0.0524 - val_rpn_bbox_loss: 0.7916 - val_mrcnn_class_loss: 0.6357 - val_mrcnn_bbox_loss: 0.3921 - val_mrcnn_mask_loss: 0.6437
Epoch 42/60
100/100 [==============================] - 115s 1s/step - loss: 0.3205 - rpn_class_loss: 0.0130 - rpn_bbox_loss: 0.0464 - mrcnn_class_loss: 0.0424 - mrcnn_bbox_loss: 0.0591 - mrcnn_mask_loss: 0.1596 - val_loss: 2.4002 - val_rpn_class_loss: 0.0505 - val_rpn_bbox_loss: 0.8065 - val_mrcnn_class_loss: 0.5787 - val_mrcnn_bbox_loss: 0.3748 - val_mrcnn_mask_loss: 0.5897
Epoch 43/60
100/100 [==============================] - 111s 1s/step - loss: 0.3398 - rpn_class_loss: 0.0123 - rpn_bbox_loss: 0.0463 - mrcnn_class_loss: 0.0508 - mrcnn_bbox_loss: 0.0655 - mrcnn_mask_loss: 0.1649 - val_loss: 2.4365 - val_rpn_class_loss: 0.0499 - val_rpn_bbox_loss: 0.7603 - val_mrcnn_class_loss: 0.6666 - val_mrcnn_bbox_loss: 0.3611 - val_mrcnn_mask_loss: 0.5986
Epoch 44/60
100/100 [==============================] - 130s 1s/step - loss: 0.3278 - rpn_class_loss: 0.0127 - rpn_bbox_loss: 0.0441 - mrcnn_class_loss: 0.0353 - mrcnn_bbox_loss: 0.0615 - mrcnn_mask_loss: 0.1743 - val_loss: 2.6866 - val_rpn_class_loss: 0.0599 - val_rpn_bbox_loss: 0.8186 - val_mrcnn_class_loss: 0.7682 - val_mrcnn_bbox_loss: 0.3911 - val_mrcnn_mask_loss: 0.6488
Epoch 45/60
100/100 [==============================] - 105s 1s/step - loss: 0.3273 - rpn_class_loss: 0.0108 - rpn_bbox_loss: 0.0442 - mrcnn_class_loss: 0.0428 - mrcnn_bbox_loss: 0.0634 - mrcnn_mask_loss: 0.1661 - val_loss: 2.4878 - val_rpn_class_loss: 0.0524 - val_rpn_bbox_loss: 0.8045 - val_mrcnn_class_loss: 0.6832 - val_mrcnn_bbox_loss: 0.3676 - val_mrcnn_mask_loss: 0.5801
Epoch 46/60
100/100 [==============================] - 107s 1s/step - loss: 0.3349 - rpn_class_loss: 0.0122 - rpn_bbox_loss: 0.0492 - mrcnn_class_loss: 0.0396 - mrcnn_bbox_loss: 0.0669 - mrcnn_mask_loss: 0.1670 - val_loss: 2.5904 - val_rpn_class_loss: 0.0453 - val_rpn_bbox_loss: 0.7682 - val_mrcnn_class_loss: 0.6614 - val_mrcnn_bbox_loss: 0.4059 - val_mrcnn_mask_loss: 0.7095
Epoch 47/60
100/100 [==============================] - 123s 1s/step - loss: 0.3280 - rpn_class_loss: 0.0108 - rpn_bbox_loss: 0.0451 - mrcnn_class_loss: 0.0513 - mrcnn_bbox_loss: 0.0613 - mrcnn_mask_loss: 0.1594 - val_loss: 2.4143 - val_rpn_class_loss: 0.0673 - val_rpn_bbox_loss: 0.8435 - val_mrcnn_class_loss: 0.5215 - val_mrcnn_bbox_loss: 0.3595 - val_mrcnn_mask_loss: 0.6224
Epoch 48/60
100/100 [==============================] - 110s 1s/step - loss: 0.3024 - rpn_class_loss: 0.0109 - rpn_bbox_loss: 0.0390 - mrcnn_class_loss: 0.0382 - mrcnn_bbox_loss: 0.0561 - mrcnn_mask_loss: 0.1582 - val_loss: 2.7538 - val_rpn_class_loss: 0.0550 - val_rpn_bbox_loss: 0.8433 - val_mrcnn_class_loss: 0.7920 - val_mrcnn_bbox_loss: 0.3944 - val_mrcnn_mask_loss: 0.6690
Epoch 49/60
100/100 [==============================] - 116s 1s/step - loss: 0.3121 - rpn_class_loss: 0.0108 - rpn_bbox_loss: 0.0481 - mrcnn_class_loss: 0.0339 - mrcnn_bbox_loss: 0.0605 - mrcnn_mask_loss: 0.1588 - val_loss: 2.6167 - val_rpn_class_loss: 0.0513 - val_rpn_bbox_loss: 0.7709 - val_mrcnn_class_loss: 0.7575 - val_mrcnn_bbox_loss: 0.3679 - val_mrcnn_mask_loss: 0.6691
Epoch 50/60
100/100 [==============================] - 108s 1s/step - loss: 0.3032 - rpn_class_loss: 0.0106 - rpn_bbox_loss: 0.0431 - mrcnn_class_loss: 0.0389 - mrcnn_bbox_loss: 0.0539 - mrcnn_mask_loss: 0.1567 - val_loss: 2.5041 - val_rpn_class_loss: 0.0445 - val_rpn_bbox_loss: 0.7563 - val_mrcnn_class_loss: 0.6547 - val_mrcnn_bbox_loss: 0.3923 - val_mrcnn_mask_loss: 0.6562
Epoch 51/60
 99/100 [============================>.] - ETA: 0s - loss: 0.2933 - rpn_class_loss: 0.0105 - rpn_bbox_loss: 0.0409 - mrcnn_class_loss: 0.0384 - mrcnn_bbox_loss: 0.0526 - mrcnn_mask_loss: 0.1510'''


if __name__ == '__main__':
	lines = TRAINING_STRING.split('\n')
	lines = list(filter(lambda x: x !='', lines))
	eval_df = pd.DataFrame(columns=['epoch'])
	epoch_row = {}
	for line in lines:
		if 'Epoch' in line:
			epoch_row['epoch'] = line.split()[-1]
		else:
			metrices = line.split(' - ')
			epoch_row['epoch_time'] = metrices[1]
			for metric in metrices[2:]:
				metric_name = metric.split(':')[0].strip()
				metric_value = metric.split(':')[1].strip()
				epoch_row[metric_name] = metric_value
			eval_df = eval_df.append(epoch_row, ignore_index=True)
			epoch_row = {}

	eval_df[['epoch','val_loss']].sort_values(by=['val_loss'], ascending=True).to_csv('Epoch_vs_loss.tsv', index=False, sep='\t')