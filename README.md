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
$ sudo apt update
#Use the following command to install pip for Python 3:
$ sudo apt install python3-pip
#Once the installation is complete, verify the installation by checking the pip version:
$ pip3 --version

```
then you can follow the commands bellow to install the service and run it :

```bash
#first download repo from github and unzip.
$ wget https://github.com/AntMouss/Novelties-detection-project/archive/master.zip -O novelties-detection-master.zip
$ unzip novelties-detection-master.zip

#changes current directory to novelties-detection-master.
$ cd novelties-detection-master

#create service environnement
python3 -m venv ./venv

#activate environment
$ source ./venv/bin/activate

#install dependencies with pip
$ pip install -r requirements.txt

# set OUTPUT_PATH environment variable to activate writing mode and have the service persistent
# else ignore this command
$ export OUTPUT_PATH=<output_path>

#launch the service
$ python server.py

```
if you don't specify `output_path` the collect service will not be persistent.

### With Docker

```bash
# build image from github repo.
$ docker build --tag novelties-detection-image https://github.com/AntMouss/Novelties-detection-project.git

# run container from the image that we build previously with creating volume that contain collect data (persistence activate) .
$ docker run -d -p 5000:5000 \
  --name <container_name> \
  --mount source=<volume_name>,target=/collect_data \
  -e OUTPUT_PATH=/collect_data \
  novelties-detection-image:latest
  
# if you don't want persistence activate , you need to pass this command instead without volume name.
$ docker run -d -p 5000:5000 --name <container_name> novelties-detection-image:latest

# to check the logs from the container ,
# use this command with the same container_name of the command above.
$ docker logs <container_name>

# you can access the volume data with this command if you are on Ubuntu with sudo privilege.
$ sudo ls /var/lib/docker/volumes/<volume_name>/_data

```
The service run on port 5000 so make sure there isn't other application running on this port before launching.