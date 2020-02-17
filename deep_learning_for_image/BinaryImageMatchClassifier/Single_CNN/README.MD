<h1>Binary Images Classification</h1>
<p>This project is used to train a simple deep learning module for image matching.</p>

<h3>Prerequisities</h3>
<p>The module was developed in python3 with usage of numpy, opencv, pillow and keras libraries.</p>


<h3>Start Training</h3>
<p>Following Step will suffice to train the module and generate a model</p>
<pre>
<code>python Binary_Trainer.py --input_img_folder=images_for_training </code></pre>
<p>The code will generate binary image match classification model.
</p>
 <strong>Command Line Arguments:</strong>
<ul>
    <li><strong>input_img_folder: </strong>Image Directory for training</li>
</ul>


<br/>
<h3>Start Predicting</h3>
<pre>
<code>python Binary_Predictor.py --image_1=./predict_images/1.jpg --image_2=./predict_images/2.jpg</code></pre>
<p>Provides match/unmatch binary classification along with confidence score.
</p>
 <strong>Command Line Arguments:</strong>
<ul>
    <li><strong>image_1: </strong>Image path.</li>
  <li><strong>image_2: </strong>Image path.</li>
</ul>


<h2>Built With</h2>
<ul>
<li>Python3</li>
<li>python-numpy</li>
  <li>python-pandas</li>
<li>Keras</li>
</ul>

<h2>Author</h2>
<ul>
<li><strong>Lava Kafle</strong></li>
<li><strong>Rohan Man Amatya</strong></li>
</ul>