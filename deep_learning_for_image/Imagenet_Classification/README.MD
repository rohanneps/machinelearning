<h1>Imagenet</h1>
<p>Use of pre-trainned models for image classification.</p>
<h3>Prerequisities</h3>
<p>The module was developed in python3 with usage of numpy, keras and other libraries.</p>
<h3>Start Predicting</h3>
<p>Following Step will suffice to predict an image.</p>
<pre><code>python imagenet_batch_prediction.py  --model=vgg19 --img_dir_path=images</code></pre>
<p><strong>Command Line Arguments:</strong></p>
<ul>
<li><strong>img_path</strong> = Image Path containing image to predict</li>
</ul>

<p><strong>Available models:</strong></p>
<ul>
<li>vgg16</li>
<li>vgg19</li>
<li>resnet</li>
<li>inception</li>
</ul>
<h2>Built With</h2>
<ul>
<li>Python3</li>
<li>argParse</li>
<li>Keras</li>
<li>python-numpy</li>
</ul>
<h2>Author</h2>
<ul>
<li><strong>Rohan Man Amatya</strong></li>
</ul>