# Novelties Detection

Novelties Detection project is a **real-time automatic newspaper semantic analyser** service , The project purpose is to understand information spreading in real-time inside a news flow .
The news provide from different rss feed source like influent newspaper  , influent news websites ( ie : New-York Times  , Fox news ...). The basic features of the service is to recognize topic contain in news feed using **topic modeling¹** approach
then we can detect what topics are a novelties or habits  , what topic appear or disappear at each time window ...

_Note_ : The default service settings are in **French** and the **rss feed** are French Source information . But you can set your own source information as explain in this [section](#1rss-feed-configuration)

## How the service works.

The service works as a two-hand service:

* First , the service collect data from various newspaper on the web in real-time
  with rss feeds that contain information about new articles posted every moment by the newspaper website.
  If you want to learn more about **rss feed** usage ,  see [here](https://en.wikipedia.org/wiki/RSS).
  _this process is repeat every **N** minutes as referred in the [base schema](#basic-architecture-schema) bellow _.
* Second ,  we apply topic model method on articles corpus that return keywords relationships and main topics contain in the current corpus (in our case the articles collected in the considered time window).
  before we process data cleaning and text pre-processing before topic modeling operation .
  _this process is repeat every **M** minutes as referred in the [base schema](#basic-architecture-schema) bellow_.

The service analyse the articles collected by each time windows and is able to provide thematics and keywords that appear,
disappear or stable during the considered time window ( ie : the topics containing the  words "queen" , "Elisabeth" , "death" appear in the window 19h - 20h Friday 09/09/2022).
the news analysis is sequential that means that the current window of data that contain the article information of this time window is compared to the last window.

We use topic modeling methods to get the words clusters that represent thematics (topics) with strong relationship in our window data, and we can compute the similarity between 2 consecutive windows using Jaccard similarity.

### Basic architecture schema:

![A test image](src/diagram/main_diagram.png)

for each blue point you can refer to the section [explanation](#explanation) , this will help you to understand how to configure custom novelties-detection service.

## Installation and Execution

### Prerequisite

if you are on Ubuntu 20.04 , you can follow the "[with shell](#with-shell)" installation section.
else you can use docker , referred to this [section](#with-docker) but first you need to install docker-engine on
your machine . The installation steps of docker engine for your operating systems might be slightly different,
please refer to the [docker documentation](https://docs.docker.com/engine/install/) for details.

### with Shell

make sure that you have pip for python 3.8 install on your machine else you can use the following commands
for pip installation:

```bash
#Start by updating the package list using the following command:
sudo apt update
#Use the following command to install pip for Python 3:
sudo apt install python3-pip
#Once the installation is complete, verify the installation by checking the pip version:
pip3 --version

```

then you can follow the commands bellow to install the service and run it :

```bash
#first download repo from github and unzip.
wget https://github.com/AntMouss/Novelties-detection-project/archive/master.zip -O novelties-detection-master.zip
unzip novelties-detection-master.zip

#changes current directory.
cd Novelties-detection-project

#create service environnement
python3 -m venv ./venv

#activate environment
source ./venv/bin/activate

#install dependencies with pip
pip install -r requirements.txt

# set OUTPUT_PATH environment variable to activate writing mode and have the service persistent
# else ignore this command
export OUTPUT_PATH=<output_path>
```

Now you can run the server with the default settings, or you can set your own settings overwriting the `config/server_settings.py` file.
see [here](#settings) for more details about the server settings.

```bash
#launch the server
python3 server.py
```

If you don't specify `output_path` the collect service will not be **persistent²**.

### with Docker

You can build the image directly from this GitHub directory using the following command,
but you can set your own settings in this way.

```bash
# build image from github repo.
docker build --tag novelties-detection-image https://github.com/AntMouss/Novelties-detection-project.git#main
```

to use your own server settings you need to download the repository and overwrite the `config/server_settings.py` file.
see more [here](#settings).

Bellow the commands for downloading the repository and change current directory.

```bash
#first download repo from github and unzip.
wget https://github.com/AntMouss/Novelties-detection-project/archive/master.zip -O novelties-detection-master.zip
unzip novelties-detection-master.zip

#change current directory.
cd Novelties-detection-project

```

Run the container with **persistent** way.

```bash
# run container from the image that we build previously with creating volume that contain collect data (persistence activate) .
docker run -d -p 5000:5000 \
--name <container_name> \
--mount source=<volume_name>,target=/collect_data \
-e OUTPUT_PATH=/collect_data \
novelties-detection-image:latest
```

or choose the no **persistent** way with the following command.

```bash
docker run -d -p 5000:5000 --name <container_name> novelties-detection-image:latest
```

Then you can check the logs of the sever to check is everything is OK , or navigate in the volume if you activate persistent way.
The server run locally on all address with port **5000** of your machine ,
you can see the api documentation at this link: *http://127.0.0.1:5000/api/v1/*

```bash

# to check the logs from the container ,
# use this command with the same container_name of the command above.
docker logs <container_name>

# you can access the volume data with this command if you are on Ubuntu with sudo privilege.
sudo ls /var/lib/docker/volumes/<volume_name>/_data
```

*Note* : * The service run on port **5000** so make sure there isn't other application running on this port before launching.

* provide about **10-20 GB** disk space for 3 months of collect with images collection (depends on your settings).
* provide **1.5 GB** disk space for novelties-detection-image.

## Explanation

1. [Rss feed configuration](#1rss-feed-configuration)
2. [Article HTML cleaning](#2article-html-cleaning)
3. [Text Pre-processing](#3text-pre-processing)
4. [Topic Modelling](#4topic-modelling)
5. [Window Similarity computation](#5window-similarity-computation)
6. [API](#6api)

### 1.RSS feed configuration

rss feed are perfect data source for fetch information about articles in real-time (i.e publication date  , title , author name , label).
the rss plugging is handled by the file `config/RSS_feeds.json` , all the rss feed addresses must be referenced in this file.

`config/RSS_feeds.json` have two main keys : "global_remove_tags" and "rss_feed_urls" , the "global_remove_tags" keys referred
to a list of global HTML tags that we want to remove for all the articles html page during the cleaning step  (see more at [cleaning section](#2article-html-cleaning))
the "rss_feed_urls" key refer to a list of rss feed item that contain "url" , "label" and "remove_tags" fields.

* url --> url of the rss feed souce
* label --> list of label related to the rss feed item , you can choose your own label , example : sport , economy ...
* remove_tags --> list of HTML tags that we want to remove particularly to this rss feed (not globally). see more [here](#2article-html-cleaning)...

rss feed item example:

```json

{
   "url" : "http://feeds.bbci.co.uk/news/world/rss.xml",
   "label" : ["sport"],
   "remove_tags" : [
      {"tag" :  "div" , "class" : "date_time_div" , "id" : "O7p469M_time_div" }, 
      {"tag" :  "span" , "class" : "author-info" , "id" : "56Ul67L_author_div" }
   ]
}
```

You can use the default `config/RSS_feeds.json` or overwrite it with your own rss feed sources following the
format describe above.

*Note*: you can add Rss source during service Runtime using [API](#6api) endpoint :  `/RSSNewsfeedSource/AddRSSFeedSource`

### 2.Article HTML cleaning

![Cleaning process diagram](src/diagram/cleaning_diagram.png)

1. at the first step , we selected and kept the **<article/>** tag
2. HTML page contain garbage information like date , author information , references to
   next articles or advertising... that aren't interesting for the topic analysis and could pollute the [Topic modeling process](#4topic-modelling).
   because we just want to keep relevant words of the subject treated in the article.
   So we make a another cleaning layer removing bad tags.

There are 2 types of bad tags:

* **global** bad tags that we fill in the "global_remove_tags" key of the `config/RSS_feeds.json` file
* **specific** bad tags that are specific to one rss feed item because articles web page have different pattern
  according to the website are they come from ( **www.theguardians.com** hasn't same html pattern than **www.nytimes.com**).
  The pattern of the article web page are different according to the rss feed they are collected from

*Note*: a tags is a dictionnary containing 3 keys:

* "tag" key is the name of the html tag ( "div" , "h1" , "p" ... ).
* "class" key is the class name of the particular tags to remove.
* "id" key is the id of the partcular tag to remove.

<ins> Tags example : </ins>

```json
{"tag" :  "div" , "class" : "date_time_div" , "id" : "O7p469M_time_div" }
```

you can add tags manually overwriting the `config/RSS_feeds.json` file to custom your cleaning process
or you can add global tags during service Runtime using [API](#6api) endpoint `/RSSNewsfeedSource/AddRSSFeedTags`

Cleaning Example:

```html
<html>
    <article>
        <h1>
            the title of the article.
        </h1>
        <header>
            <p class="date_time">
                Monday , April 2 , 2026 15:00 UTC.
            </p class="author_info">
                Marc Duchamps (Senior reporter)
            <div class="advertising">
                <div>
                     <img src="product_image.jpg" alt="product" width="500" height="600">
                     <p>
                         Buy our dumbness product !
                     </p>

                </div>

            </div>
        </header>
        <div class="real_information">
            <p>
                some information to keep after cleaning.
            </p>

        </div class="next_articles">
            <ol>
                <li>next article 1</li>
                <li>next article 2</li>
                <li>next article 3</li>
            </ol>
        <footer>
            nothing important here.
        </footer>

    </article>
</html>
```

Result after remove **global** and **specific** tags:

```html
<html>
    <article>
        <h1>
            the title of the article.
        </h1>
        <div class="real_information">
            <p>
                some information to keep after cleaning.
            </p>

    </article>
</html>
```

3. finally, we extract real article text removing all html syntax of the string:

```
"the title of the article. some information to keep after cleaning. "
```

### 3.Text Pre-processing

Our topic Modelling process isn't **multilingual**, so we had to specify a lang during the text preprocessing because
we don't want to make topic modelling with multilingual corpus because it will not be efficient.

![text pre-processing schema](src/diagram/text_preprocessing_diagram.png)

1. **lang detection** : the first step of text preprocessing is lang detection , we don't want to pre-process wrong lang text
2. **tokenization** : tokenization in NLP is a set of method that divide string text in logical element (called **token**)
   in general token are words, but it could be punctuation marker or one word could be composed of 2 tokens , example:
   ("geography" --> token1 : "geo" , token2 : "graph"). If you want to learn more about [tokenization](https://neptune.ai/blog/tokenization-in-nlp)
3. **remove specific words** : in the diagram we talk about stop words and digits , stopwords are commun words in lang vocabulary
   which bring us no special information like : "the" , "are" , "us" etc...
4. **Lemmatization** : in linguistics is the process of grouping together the inflected forms of a word so
   they can be analysed as a single item, identified by the **word's lemma**, or dictionary form.
   example : "analysed" , "analyse" , "analysing" transformed to the lemma "analysis".

### 4.Topic Modelling

![text topic-modelling schema](src/diagram/topic_modelling_diagram.png)

1. **update global dictionary** : we use a **global dictionary** as a register of token (words , punctuation , etc...) for all window corpus (set of article texts in the time window collect).
   The purpose of this dictionary is to count every token occurrences, although we can use it to filter rare token or really common token. This dictionary is updated every new window.
2. **filter words** or **filter tokens** : as explain above we remove rare words or common tokens using the global dictionary ,
   this tokens are not topics relevant . Example : misspelled words "length" or common **irrelevant** words "Monday" .
3. **BOW** (bag of words) is a simplifying model text representation using in documents classification .
   each text in the corpus is represented as a vector where each component is the number occurrences of the token in the vocabulary index.
   This model representation is more efficient for training topic model.
4. **train model** : this part is the main part of the process , we have 2 type of training : **unsupervised training** using unlabeled texts and **supervised training** using labeled texts.
   The purpose of the supervised modelling is to return change for a predefined label and follow the evolution of the label in time.
   The unsupervised way is used to detect latent topics in a bunch of articles independently of a paper categories (label) , this type of training can return us complementary information about the news composition.
   In other words , the unsupervised modelling allows us to follow the **micro topics** evolution (little topics that appears punctually in a window) while the supervised method allows us to follow big **categories** evolution (topic persisting on many windows).

We use different topic model kernel:

* unsupervised kernel : [LDA³](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) , [CoreX⁴](https://github.com/gregversteeg/corex_topic)
* supervised kernel : [TFIDF⁵](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) , Guided LDA (or semi-supervised LDA) , Guided CoreX (or semi-supervised CoreX)

**_Note 1_** : **Guided LDA** and **Guided CoreX** kernel are based on the **LDA** and **CoreX** kernel , the difference is that in the guided case
we use label **seed** words to make the training converge around the label words distribution.
In other words , we use set of words **relevant** to a certain label, and we increase the weight of this particular words
in the corpus during training to hook other strongly related words. This **seed** words work as an **anchors** .

_**Note 2**_ :  use your own seed words relative to your labels overwriting the `config/seed.json` file
with labels as keys and list of seed words (relative to the key topics) as values else you can keep the original
`config/seed.json` file.

here is a training example schema:

![training_schema](src/diagram/training_model_schema.drawio.png)

**_Note_** : we use the [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html) LDA implementation.

### 5.Window Similarity computation

explicative diagram :

![similarity computation](src/diagram/Similarity_computation_diagram.png)

1. we use similarity calculator extracting two consecutive models corresponding to two consecutive windows, and we compute [Jaccard similarity](https://pyshark.com/jaccard-similarity-and-jaccard-distance-in-python/)
   the supervised case we compute Jaccard similarity for each topic corresponding to a label and we stack similarity score in a list:
   *example* : assume the two bellow cluster words for the label i -> "sport":

```json
{"Ai" :  ["football" , "Manchester" , "united" , "devils" , "Traford" , "victory" , "goals"],
"Bi" :  ["football" , "Manchester" , "city" , "Arsenal" , "win" , "goals"]}
```

jaccard similarity formula is:
$Ji = \frac{|Ai \cap Bi|}{|Ai \cup Bi|} = \frac{|Ai \cap Bi|}{|Ai| + |Bi| – |Ai \cup Bi|}$

here:

$Ai \cap Bi = \{"football" , "Manchester" , "goals"\}$
$|Ai \cap Bi| = 3$

and

$Ai \cup Bi = \{"football" , "Manchester" , "united" , "devils" , "Traford" , "victory" , "goals" , "city" , "Arsenal" , "win"\}$
$|Ai \cup Bi| = 10$

finally

$Ji =\dfrac{3}{10}$

2. then ,  we standardize the similarity scores to obtain the final similarity score between the window n and n-1 :

$J = \dfrac{\sum_{i=0}^{k}{Ji}}{k}$

3.the process isn't the same for unsupervised case , we append every words cluster for each topics then we compute the total jaccard similarity.

4.finally , we will classifie the change rate between two windows using normal distribution classifier .
The final result is a range percentiles :

In our case , a percentile is a similarity score below which a given percentage k of scores in its frequency distribution falls (exclusive definition) or a score at or below which a given percentage falls (inclusive definition).
For example, the 90th percentile is the similarity score below which (exclusive) or at or below which (inclusive) 90% of the scores in the distribution may be found:

![percentile-normal-curve](src/normal_percentile.png)

We classifie similarity between window as percentile ranges which means a window similarity could be for example 1-5% or 5-20% ... rarely high or low:
example of percentile ranges:

![percentile-ranges](src/normal_percentile2.png)

_**Note**_ : we use normal distribution because we previously analyse the distribution of our similarity calculator that fit normal distribution

### 6.API

(Not available yet but you can see query swagger doc at https://127.0.0.1:5000/api/v1 )

## Server Settings

this section will help you to custom the service , most of the settings refer to the [explanation](#explanation) section.
You need to overwrite the `config/server_settings.py` file else you can keep the default settings.

### collect settings

* `LOOP_DELAY_COLLECT` : delay between 2 collect process corresponding to the N value in the main [schema](#basic-architecture-schema) (in minutes)
* `COLLECT_RSS_IMAGES` : boolean control the collect of images in the rss feed (if True you need to specify `OUTPUT_PATH`)
* `COLLECT_ARTICLE_IMAGES` : boolean control the collect of images in the article page.html (if True you need to specify `OUTPUT_PATH`)
* `COLLECT_HTML_ARTICLE_PAGE` : boolean control the collect of the html article (if True you need to specify `OUTPUT_PATH`)
* `PRINT_LOG` : boolean control log of the collect process (performance and error)

**_Note_** : the collect process can write data in fileSystem if you used the [persistent](#installation-and-execution) mode (specify `OUTPUT_PATH`)
so you can set `COLLECT_RSS_IMAGES` , `COLLECT_ARTICLE_IMAGES` and `COLLECT_HTML_ARTICLE_PAGE` as True else it return an exception.

### labels idx settings.

* `LABELS_IDX` : list of targeting label .
  **_Warning_** : You can't change `LABELS_IDX` during runtime, you can't add new label because we want to keep label traceability.

### Process window settings

* `LOOP_DELAY_PROCESS` : delay between 2 Windows processing corresponding to the M value in the main [schema](#basic-architecture-schema) (in minutes)
* `MEMORY_LENGTH` : integer --> number of window keep in memory (can't exceed 30)

**_Note_** : `LOOP_DELAY_PROCESS` must be superior to `LOOP_DELAY_COLLECT` because the processing need data collecting first
else it returns an exception.

### Text pre-processing settings

* `LANG` : lang code of the pre-processed texts (can't pre-process text in other lang because the service isn't multilingual)
* `LEMMATIZE` : boolean control the lemmatization
* `REMOVE_STOP_WORDS` : boolean control the stop words removing
* `REMOVE_NUMBERS` : boolean control the numbers removing
* `REMOVE_SMALL_WORDS` : boolean control the small words removing

**_Note_** : you can specify the minimum length of a "small words" or "tall words" and you can add a predefine list of words to remove
follow the example:

```python
# TEX PRE-PROCESSOR SETTINGS
LANG: str = "en"
LEMMATIZE: bool = False
REMOVE_STOP_WORDS: bool = True
REMOVE_NUMBERS: bool = True
REMOVE_SMALL_WORDS: bool = True
SMALL_WORDS_SIZE = 3
my_undesirable_words = ["Monday", "March", "follow", "Twitter"]

PREPROCESSOR = MetaTextPreProcessor(
    lang=LANG,
    lemmatize=LEMMATIZE,
    remove_stop_words=REMOVE_STOP_WORDS,
    remove_numbers=REMOVE_NUMBERS,
    remove_small_words=REMOVE_SMALL_WORDS,
    min_word_size=SMALL_WORDS_SIZE,
    undesirable_words=my_undesirable_words
)
```

### Macro-calculator settings (**supervised calculator**)

Customise Macro-calculator settings overwriting the **#MACRO-CALCULATOR SETTINGS**

* `MACRO_CALCULATOR_TYPE` : Type of the Macro-Calculator.
* `macro_training_args` : dictionary of arguments relatives to the training of the kernel.
* `macro_kwargs_results` : dictionary of arguments relatives to the similarity computation between windows.

There are 3 types of Macro Calculator :

* `TFIDFSequentialSimilarityCalculator` : using a TFIDF kernel (no specific training arguments available yet).
* `GuidedCoreXSequentialSimilarityCalculator` :  using a CoreX kernel ( specific training arguments in original [doc](https://github.com/gregversteeg/corex_topic/blob/master/corextopic/corextopic.py) line 18 ).
  _**Warning**_ : you can't use following parameters --> `n_hidden` .
* `GuidedLDASequentialSimilarityCalculator` :  using LDA kernel ( specific training arguments in original [doc](https://radimrehurek.com/gensim/models/ldamodel.html) ).
  _**Warning**_ : you can't use following parameters -->  `corpus`, `num_topics`, `id2word` , `eta` .

_**Note** 1_ : as explain in [note 2](#4topic-modelling) , you need to specify seed words if you use semi-supervising learning
( `GuidedLDASequentialSimilarityCalculator` , `GuidedCoreXSequentialSimilarityCalculator` ) else you can keep default seed words in `config/seed.json`
( just available for default `LANG` and default `LABELS_IDX` )

_**Note 2**_ : If you are using the 2 last class , you can specify an `anchor_strength` training parameter (semi-supervising learning)
to increase the weight of the seed words in the corpus , else you can keep the default `anchor_strength` value.

**_Warning_** : Do not confuse seed words parameters with `seed` parameters in `GuidedCoreXSequentialSimilarityCalculator` instance
that enabled reproducible training.

Example of Macro-Calculator settings:

```python
# MACRO-CALCULATOR SETTINGS
MACRO_CALCULATOR_TYPE : type = Sequential_Module.GuidedLDASequentialSimilarityCalculator
macro_training_args = {
    "anchor_strength" : 100,
    "passes" : 3 ,
    "minimum_probability" : 0.2
}
macro_kwargs_results : dict = {
    "ntop" : 100,
    "remove_seed_words" : True,
    "back"  :  3
}
```

### Micro-calculator settings (**unsupervised calculator**)

Customise Micro-calculator settings overwriting the **#MICRO-CALCULATOR SETTINGS**

* `MICRO_CALCULATOR_TYPE` : Type of the Micro-Calculator.
* `micro_training_args` : dictionary of arguments relatives to the training of the kernel.
* `NB_MI_TOPICS` : integer referring to the number of topics in each window (this number is fix for each window).

There are 2 types of Micro-Calculator :

* `CoreXSequentialSimilarityCalculatorFixed` :  using a CoreX kernel ( specific training arguments in original doc ).
  _**Warning**_ : you can't use the following parameters -->  `n_hidden` .
* `LDASequentialSimilarityCalculatorFixed` :  using LDA kernel ( specific training arguments in original doc ).
  _**Warning**_ : you can't use the following parameters -->  `corpus`, `num_topics`, `id2word` .

Example of Micro-Calculator settings:

```python
# MICRO-CALCULATOR SETTINGS
NB_MI_TOPICS = 7
MICRO_CALCULATOR_TYPE : type = Sequential_Module.GuidedCoreXSequentialSimilarityCalculator
micro_training_args = {
    "anchor_strength" : 4,
    "tree" : False
}
```

### bad words removing settings

* `fct_above` : Function which defines the maximum number of document appearances for a word otherwise it is removed from the original vocabulary.
* `fct_bellow` : Function which defines the minimum number of document appearances for a word otherwise it is removed from the original vocabulary.
* `kwargs_above` : Dictionary contain `fct_above` specific arguments .
* `kwargs_bellow` : Dictionary contain `fct_bellow` specific arguments .

There are 3 types of filtering functions :

* `absoluteThresholding` --> linear function with $slop = 0$
  example : with $intercept = 100$
  ![absolute_thresholding](src/abs_figure.png)
**Document appearances** --> number of documents (articles in our case) in which the considerate token (word) is present. 
**Above-removing area** --> if a word is in this area the fct_above function will remove it from the original vocabulary during [the tokens filtering](#4topic-modelling),
so it will be not considerate at the topic-modelling step .
**Below-removing area** --> if a word is in this area the fct_below function will remove it from the original vocabulary during [the tokens filtering](#4topic-modelling),
so it will be not considerate at the topic-modelling step .
* `linearThresholding` --> linear function with $ 0 <= slop <= 1$
  example : with $slop = 0.25$ and $intercept = 0$
  ![linear_thresholding](src/rel_figure.png)
* `logarithmThresholding` --> exponential reverse function with 2 arguments : `limit`
  example : $limit = 0.6$
  ![logarithm_thresholding](src/log_figure.png)


## Lexical

1. Topic modeling : Set of machine learning method which main purpose are to found
   relationship between words in a document corpus in order to return words clusters (topic).
2. Persistent : The persistent "mode" save the collect data to the `output_path` directory.
3. LDA : Latent Dirichlet Allocation.
4. TFIDF : term frequency–inverse document frequency.
5. CoreX : correlation explanation.
