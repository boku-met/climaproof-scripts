# climaproof-scripts
This repository contains the scripts used for calculating climate indicators and a user guide.

# ****** HOW TO USE THE SCRIPTS ******

## Welcome!
In this repository (https://github.com/boku-met/climaproof-scripts) you can find the scripts used to calculate climate indicators in the ClimaProof project. The programming language used is Python. In order to run the scripts and calculate indicators, you need to follow a few steps that are listed in this file.

All climate indicators have been published for all available climate scenarios of the entire Western Balkan region:

https://data.ccca.ac.at/group/ec12144b-a8f1-40bc-9297-931bcfd01b5a?groups=ci

So there is only reason to use these scripts if you have customised the climate scenarios by improving data resolution (downscaling tool) and/or data quality with observations (ICC-OBS tool).  

## Step 1: Download the scripts
If you already know which indicator you want to calculate, you have several options of downloading the scripts:

- Click on the name of the script you are interested in and copy the code into an empty text file. You can then save the file under "name_of_the_script.py"
- Download all scripts by clicking on the green "Code" icon, and then "Download ZIP"
- If you have git installed on your local machine, you can clone this repository with

    git clone https://github.com/boku-met/climaproof-scripts.git
  
## Step 2: Setting up the environment
The scipts are developed to run under the same conda environment as the ClimaProof tools. So if you already set up the environment described in the chapter "Installing the Python environment" in the "Handbook_UserGuide" (https://github.com/boku-met/climaproof-docs) you are good to go. If not, please do so. 

## Step 3: Get the data
Each climate indicator requires certain climate parameters. Please make sure to check the scripts to see which parameters are required:

Open script - look for text block "def user_data()" at the top - first comment specifies the required input data

You can use the ClimaProof tools to customise the available climate scenarios or download them directly:
https://data.ccca.ac.at/group/climaproof

The scripts use only one file path to look for input data, so make sure to put all data into the same folder.

## Step 4: Specify file paths and user info
Each script contains a section labeled "def user_data()" at the beginning. In this section, you can specify the path to the required input data, and a few additional options. The file paths need to be written within quotation marks (""). Each user-defined setting is explained by a comment. When you finished setting the user details, make sure to save the script text file, otherwise it will not work.
### Attention:
Starting the scripts without setting the user info will not work. Starting them with the wrong info might lead to unwanted results.
Always check that the user info is correct before you run the scripts.

## Step 5: Run the scrips
When you have the conda environment installed (Step 2), running the scripts is straightforward. In the command line, type

    conda activate tools (or user-defined name of the ClimaProof environment)
    python name_of_the_script.py

The scripts produce some text output. If you want to avoid the text in your command line, you can direct the output to a text file:

    python name_of_the_script.py > name_of_textfile.txt

The scripts can take some time if they need to process large files. You can also run the programme in the background (Linux and MacOS only):

    python name_of_the_script.py &
    python name_of_the_script.py > name_of_textfile.txt &

For a hint how to run the scripts in the background under Windows, see
https://superuser.com/questions/198525/how-can-i-execute-a-windows-command-line-in-background


### If you have any questions or run into troubles with the scripts, please contact us!
