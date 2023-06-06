# BIOSCAN-1M

###### <h3> Overview
This repository houses the codes and data pertaining to the BIOSCAN-1M project. 
Within this project, we introduce the BIOSCAN-1M Insect dataset, which can be accessed 
for download via the provided link. The repository encompasses code for data sampling and splitting, 
dataset statistics analysis, as well as image-based classification experiments centered around 
the taxonomy classification of insects. 
 
###### <h3> Dataset
BIOSCAN dataset provides researchers with information about insects. 
Three main sources of information published by the BIOSCAN dataset are: 

###### <h4> I. Biological Taxonomy Ranking 

![My Image](dataset/bioscan_images/taxonomy.png "Biological Taxonomy Ranking")

###### <h4> II. DNA Barcode Sequences and Barcode Indexing

* DNA barcode sequence
* Barcode Index Number (BIN)

###### <h4> III. RGB Images 

We publish four packages of the BIOSCAN-1M Insect images, 
each package is structured with 113 chunks of 10,000 images each:
- (1) Original JPEG images.
- (2) Cropped JPEG images.
- (3) Original JPEG images resized to 256 on the smaller dimensions.
- (4) Cropped JPEG images resized to 256 on their smaller dimension.

 <p align="middle">
  <img src="dataset/bioscan_images/3995976_Blattodea.jpg"     alt="Blattodea"     title="Blattodea" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4049775_Hemiptera.jpg"     alt="Hemiptera"     title="Hemiptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4079301_Archaeognatha.jpg" alt="Archaeognatha" title="Archaeognatha" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4079804_Psocodea.jpg"      alt="Psocodea"      title="Psocodea" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4091453_Embioptera.jpg"    alt="Embioptera"    title="Embioptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4273164_Dermaptera.jpg"    alt="Dermaptera"    title="Dermaptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4279962_Ephemeroptera.jpg" alt="Ephemeroptera" title="Ephemeroptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4284053_Odonata.jpg"       alt="Odonata"       title="Odonata" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/4285466_Plecoptera.jpg"    alt="Plecoptera"    title="Plecoptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/5071176_Thysanoptera.jpg"  alt="Thysanoptera"  title="Thysanoptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/5131549_Neuroptera.jpg"    alt="Neuroptera"    title="Neuroptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/5154627_Trichoptera.jpg"   alt="Trichoptera"   title="Trichoptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/5189695_Hymenoptera.jpg"   alt="Hymenoptera"   title="Hymenoptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/5578509_Zoraptera.jpg"     alt="Zoraptera"     title="Zoraptera" width="150" hspace="2"/>
  <img src="dataset/bioscan_images/5580278_Coleoptera.jpg"    alt="Coleoptera"    title="Coleoptera" width="150" hspace="2"/>
</p>

<p align="middle">  $$ Insect \space Orders $$ </p>

$${\color{red}Blattodea \space \space \color{blue}Hemiptera \space \space \color{orange}Archaeognatha \space  \space \color{green}Psocodea \space \space \color{purple}Embioptera}$$

$${\color{red}Dermaptera \space \space \color{blue}Ephemeroptera \space \space \color{orange}Odonata \space \space \color{green}Plecoptera \space \space \color{purple}Thysanoptera}$$
  
$${\color{red}Neuroptera \space \space \color{blue}Trichoptera \space \space \color{orange}Hymenoptera \space \space \color{green}Zoraptera \space \space \color{purple}Coleoptera}$$

###### <h3> Metadata File
In addition to the image dataset, we have also published a corresponding metadata file for our dataset, 
named "BIOSCAN_Insect_Dataset_metadata". This metadata file is available in both dataframe format (.tsv) 
and JSON-LD format (.jsonld). 
The metadata file encompasses valuable information, including taxonomy annotations, DNA barcode sequences, 
and indexes and labels for each data sample. Furthermore, the metadata file includes the image names and unique IDs 
that reference the corresponding storage location of each image. It also provides insights into the roles of the 
images within the split sets. Specifically, it indicates whether an image is used for training, validation, or 
testing in the six experiments conducted in our paper. 

To run the following steps you first need to download dataset and the metadata file, 
and make path settings appropriately.

###### <h3> Dataset Statistics
To see statistics of the dataset, and run the following:
```bash
python main.py --print_statistics 
``` 
 
To split dataset into Train, Validation and Test sets run the following:
```bash
python main.py --make_split --print_split_statistics
``` 
 
###### <h3> Classification Experiments
Two image-based classification experiments were conducted, focusing on the taxonomy ranking of insects. 
The first set of experiments involved classifying images of insects into 16 Orders. 
The second set of experiments specifically targeted the Order Diptera and 
aimed to classify its members into 40 families, which constitute a significant portion of the order.

To train the model on classification task using a baseline model run setting the name of experiments:
```bash
python main.py --loader --train --exp_name small_insect_order
``` 

###### <h3> Preprocessing
In order to enhance efficiency in terms of time and computational resources for conducting experiments 
on the BIOSCAN dataset's RGB images, we implemented an offline preprocessing step. This involved utilizing 
a cropping tool to modify the original RGB images. 
By applying this preprocessing step, we aimed to optimize the subsequent experimental processes.
To use the cropping tool, first add the module:


```bash
python main.py --crop_image
``` 


###### <h3> Requirement 
The requirements file used to run experiments is available in the requirement.txt.
  
###### <h3> Collaborators
"Nicholas Pellegrino" <nicholas.pellegrino@uwaterloo.ca> & "Ming Gong" <ming_gong@sfu.ca>  

 

 

 

