<h1>Binary Image Classification</h1>
<p>This project is used to train a binary deep learning network for Match/NotMatch image classification</p>

<h3>Prerequisities</h3>
<p>The module was developed in python3 with usage of numpy, opencv and keras libraries.</p>

<h3>Start Training</h3>
<h4>Train/Test Requirement</h4>
<p>The images under the train and test folders needs to be segregated into categories where similar type images are kept in the same category. There need to be minimum of images as configurable under config.NUM_MATCH_NOT_MATCH_PAIR_CNT</p>

<h4>Following Step will suffice to train the module and generate a model</h4>
<pre>
<code>python generate_dataset_and_train_image_generator.py</code>
</pre>

<br/>
<h3>Start Predicting</h3>
<pre>
<code>python predict.py</code></pre>
<p>Predicts whether the images match or not with confidence score.
</p>

<br/>

<h2>Built With</h2>
<ul>
<li>Python3</li>
<li>python-numpy</li>
  <li>python-opencv</li>
<li>Keras</li>
</ul>

<h2>Author</h2>
<ul>
<li><strong>Rohan Man Amatya</strong></li>
</ul>