# Novelties Detection


## About this project.


## Installation and Execution

### Prerequisite

if you are on Ubuntu 20.04 , you can follow the kit shell installation as referred [here](#with-shell) . 
else you can use docker , referred to this [section](#with-docker) but first you need to install docker-engine on
your machine . The installation steps of docker engine for your operating systems might be slightly different, please refer to the [docker documentation](https://docs.docker.com/engine/install/) for details.

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
or choose the no persistent way

```bash
docker run -d -p 5000:5000 --name <container_name> novelties-detection-image:latest
```

Then you can check the logs of the sever to check is everything is OK , or navigate in the volume if you activate persistent way.
The server run locally on all adress with port 5000 of your machine , you can see the api documentation at this link: http://127.0.0.1:5000/api/v1/ 

```bash

# to check the logs from the container ,
# use this command with the same container_name of the command above.
docker logs <container_name>

# you can access the volume data with this command if you are on Ubuntu with sudo privilege.
sudo ls /var/lib/docker/volumes/<volume_name>/_data
```
The service run on port 5000 so make sure there isn't other application running on this port before launching.


## Server Settings