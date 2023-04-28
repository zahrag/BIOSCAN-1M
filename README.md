# BioScan-1M

###### <h3> Overview
This repository contains the codes and data for BioScan-1M datasets project. 
In this project, three datasets of different sizes are published. 
The dataset files can be downloaded through the **link**. 
There are different classification experiments conducted using RGB images in this project.
 
 
###### <h3> Datasets
BioScan dataset provides researchers with information about living organisms. 
Three main sources of information published by the BioScan dataset are: 

###### <h4> I. Biological taxonomy ranking of the organisms

![My Image](dataset/bioscan_images/taxonomy.png "Biological Taxonomy Ranking")

###### <h4> II. DNA barcode sequences and barcode indexing

* DNA sequence
* Barcode Index Number (BIN)

###### <h4> III. RGB images of the individual organisms

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


###### <h3> BioScan Subsets
To facilitate different levels of computational processing, we publish three varying sizes of the dataset: 

* **BioScan-80K**: Small size dataset with 82,728 data samples.
* **BioScan-200K**: Medium size dataset with 195,585.
* **BioScan-1M**: Large size dataset with 1,285,378.

Due to limited space, there are only metadata files of the small dataset (BioScan-80K) and its train, 
validation and test splits available in dataset folder together with a small set of RGB images.
 
###### <h3> Datset Statistics

To see statistics of the small dataset run the following:
```bash
python main.py --print_statistics 
``` 
 
To split the small dataset into Train, Validation and Test sets run the follwoing:
```bash
python main.py --make_split --print_split_statistics
``` 
 
###### <h3> BioScan Classification 
We conducted multi-class classification experiments using RGB images of the BioScan datasets. 
We addressed two classification problems based on biological taxonomy ranking annotation 
available by the BioScan dataset.

* Insect-Class Order-Level image classification: In total 16 different **orders** of insects are predicted.
* Insect-Class Order-Diptera image classification: In total 40 **families** of the Diptera from insect class are predicted.  

To train the model on classification task using a baseline model run:
```bash
python main.py --loader --train
``` 

###### <h3> Preprocessing
To increase efficiency with respect to time and computational resources required for running experiments 
with RGB images of the BioScan dataset, we performed offline preprocessing by applying a cropping tool 
on the original RGB images. 

To use the cropping tool, first add the module:

```bash
git submodule add git@github.com:zmgong/BioScan-croptool.git crop_tool --force
``` 

```bash
python main.py --crop_image
``` 


###### <h3> Requirement 
The requirements file used to run experiments is available in the requirement.txt.
  
###### <h3> Collaborators
"Nicholas Pellegrino" <nicholas.pellegrino@uwaterloo.ca> & "Ming Gong" <ming_gong@sfu.ca>  

 

 

 

