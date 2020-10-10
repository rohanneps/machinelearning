<h1>Instance Segmentation for Farm Dams</h1>
<p>This project is used to perform transfer learning on instance segmentation to identify farm dams and their corresponding pixel-wise occupancy from satellite images.</p>

<h3>Prerequisities</h3>
<p>The module was developed in python3 with usage of numpy, opencv, pillow, tensorflow and keras libraries.</p>


<h3>Start Training</h3>
<p>Following notebook will train(transfer learn existing model) model and generate a models for inference.</p>
<pre>
<code>train.ipynb</code></pre>
<p>The code will generate segmentation model.
</p>

<br/>
<h3>Start Predicting</h3>
<pre>
<code>prediction.ipynb</code></pre>
<p>Provides farm dam identification and mask prediction.</p>
 


<h2>Built With</h2>
<ul>
<li>Python3</li>
<li>python-numpy</li>
  <li>python-pandas</li>
<li>tensorflow</li>
<li>opencv</li>
<li>Keras</li>
</ul>

<h2>Citation</h2>
<pre>@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
</pre>

<h2>Acknowledgements</h2>
<p>
The codes are based on implementation of Mask R-CNN by (https://github.com/matterport/Mask_RCNN). The model generates mask region and segmentation masks for farm dams in the statellite image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.
</p>

<h2>Author</h2>
<ul>
<li><strong>Rohan Man Amatya</strong></li>
</ul>