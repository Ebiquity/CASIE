# CASIE : CyberAttack Sensing and Information Extraction

## Tool for extracting cyberattack event information 
Components of CASIE are written in Python:
1. nug\_arg\_detection.py (applied Domain-Word2vec) or nug\_arg\_detection_bert.py (applied BERT) are for training the neural classifier to detect nugget and argument. 
2. realis_identify.py is for training the neural classifier to identify realis for event nugget.
3. role_phrase.py is for training the neural classifier to assign roles to arguments.
4. link_coref.py is for identifying realis and linking role using in the trained models and grouping the mentions of the coreference events.
5. Others scripts are needed for running CASIE's components.

## Annotation corpus of cybersecurity event in news articles
The corpus contains 1000 annotation and source files.
Our cybersecurity focused on five event types: Databreach, Phishing, Ransom, Discover, and Patch.

More details of the annotation and CASIE's system are in the papers. If you use our data, please cite one of the following papers.
1. Taneeya Satyapanich, Francis Ferraro, and Tim Finin, "CASIE: Extracting Cybersecurity Event Information from Text", InProceedings, Proceeding of the 34th AAAI Conference on Artificial Intelligence, February 2020.
2. Taneeya Satyapanich, Tim Finin, and Francis Ferraro, "Extracting Rich Semantic Information about Cybersecurity Events", InProceedings, Second Workshop on Big Data for CyberSecurity, held in conjunction with the IEEE Int. Conf. on Big Data, December 2019.


Any problems found, please contact taneeya1@umbc.edu.

