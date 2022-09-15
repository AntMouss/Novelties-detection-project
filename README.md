# Novelties Detection


Novelties Detection project is a **real-time automatic newspaper semantic analyser** that provide you keywords and set of keywords for different thematics
inside newspaper data .
The purpose of the project is to have better understanding of novelties in actuality and detect when actuality change appear.


## Explanation about the project.

Firstly the service collect data from various newspaper on the web in real-time
thanks to rss feeds that contain information about new articles posted every moment by the paper company which handdle this rss feed server.
If you want know more about rss feed usage ,  see [here](https://en.wikipedia.org/wiki/RSS).

Secondly we want to use texts from the articles to apply topic model algorithm that return us keywords and thematics contain in the corpus (in our case the articles collected in the considered time window).
so we apply data cleaning and text pre-processing before processing.

Finally, the service analyse the articles collected and is able to provide thematics and keywords that appear,
disappear or stable during a certain time window ( ex : the words queen , Elisabeth , death appear in the window 19h - 20h Friday 09/09/2022).
the news analysis is sequential that means that the current window of data that contain the article information of this time window is compared to the last window that contain article information of the last time window.
We use topic model method to get the words clusters that represent thematic with strong relationship in our window of data and we can calculate the similarity between 2 consecutive windows using jaccard distance.

Architecture system schema:
```xml
<mxfile host="app.diagrams.net" modified="2022-09-13T09:58:29.429Z" agent="5.0 (X11)" etag="kGURadZfN5nUiKLP8WWM" version="20.3.0" type="google"><diagram name="Page-1" id="5f0bae14-7c28-e335-631c-24af17079c00">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</diagram></mxfile>
```


## Installation and Execution

### Prerequisite

if you are on Ubuntu 20.04 , you can follow the kit shell installation as referred [here](#with-shell) . 
else you can use docker , referred to this [section](#with-docker) but first you need to install docker-engine on
your machine . The installation steps of docker engine for your operating systems might be slightly different,
please refer to the [docker documentation](https://docs.docker.com/engine/install/) for details.

### With Shell

make sure that you have pip for python3 install on your machine else you can use the following commands
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
Now you can run the server with the default settings or you can set your own settings overwritting the `config/server_settings.py` file.
see [here](#settings) for more details about the server settings.

```bash
#launch the server
python3 server.py
```

If you don't specify `output_path` the collect service will not be persistent.

### With Docker

You can build the image directly from this github directory using the following command,
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

Run the container with persistent way.

```bash
# run container from the image that we build previously with creating volume that contain collect data (persistence activate) .
docker run -d -p 5000:5000 \
--name <container_name> \
--mount source=<volume_name>,target=/collect_data \
-e OUTPUT_PATH=/collect_data \
novelties-detection-image:latest
```

or choose the no persistent way with the following command.

```bash
docker run -d -p 5000:5000 --name <container_name> novelties-detection-image:latest
```

Then you can check the logs of the sever to check is everything is OK , or navigate in the volume if you activate persistent way.
The server run locally on all adress with port 5000 of your machine ,
you can see the api documentation at this link: http://127.0.0.1:5000/api/v1/ 

```bash

# to check the logs from the container ,
# use this command with the same container_name of the command above.
docker logs <container_name>

# you can access the volume data with this command if you are on Ubuntu with sudo privilege.
sudo ls /var/lib/docker/volumes/<volume_name>/_data
```
The service run on port 5000 so make sure there isn't other application running on this port before launching.


## Server Settings