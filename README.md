# Open-Stereotype-corpus

This repository contains **Open-Stereotype**, a corpus annotated using a bottom-up taxonomy. The Corpus has been presented in the paper:

**Subjectivity in Stereotypes Against Migrants in Italian: An Experimental Annotation Procedure**, accepted at Clic-It 2025. 


## 📝 Abstract
The presence of social stereotypes in NLP resources is an emerging topic that challenges traditionally used approaches for the creation of corpora and resources. An increasing number of scholars proposed strategies for considering annotators' subjectivity in order to reduce such bias both in computational resources and in NLP models. In this paper, we present Open-Stereotype, an annotated corpus of Italian tweets and news headlines regarding immigration in Italy developed through an experimental procedure for the annotation of stereotypes aimed to investigate their different interpretation. The annotation is the result of a six-step process, where annotators identify text-spans expressing stereotypes, generate rationales about these spans and group them in a more comprehensive set of labels.
Results show that humans exhibit high subjectivity in conceptualizing this phenomenon, and that the prior knowledge of an Italian LLM leads to more consistent classifications of specific labels that do not depend on annotators' background. 

## 🔧 Annotation pipeline

![](<O-Ster dataset/original_dataset/annotation_pipeline.png>)


## ℹ️ Columns description

### annotation process
- *id*: id that uniquely identifies each tweet
- *annotatore*: id that uniquely identifies each annotator 
- *tweet*: tweet that was annotated
- *chunk*: span of text that explicitely conveyed stereotypical content (Phase 2 - Identifying spans)
- *annotazione*: rationale that explicitely expresses the sense behind the stereotype and the target group (Phase 3 - Rationales)
- *annotazioni_parsate*: processed rationales. (Phase 4 - Text processing)
- *cluster_10_nome_{annotator_number}*: descriptive labels assigned by each annotator(Phase 5 - Free Text labeling)
- *cluster_5_nome_{annotator_number}*: group of labels defined by each annotator(Phase 6 - Grouping)
- *cluster_10_{annotator_number}*: random numerical number assigned to the descriptive labels
- *cluster_5_{annotator_number}*: random numerical number assigned to the groups

### From HaSpeeDe2
The following columns provide information about the HaSpeeDe2 corpus from which the dataset was extracted 

*M. Sanguinetti, G. Comandini, E. Di Nuovo, S. Frenda, M. A. Stranisci, C. Bosco, C. Tommaso, V. Patti, R. Irene, HaSpeeDe 2 @ EVALITA2020: Overview of the EVALITA 2020 Hate Speech Detection Task, in: EVALITA 2020 Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian, CEUR, 2020, pp. 1–9.*

- time
- source
- text_original
- note
- set
- id_original
- text
- target
- hs
- stereotype

### Additional dimensions
The following columns represent the other dimensions of annotations added in the work by *S. Frenda, V. Patti, P. Rosso, Killing me softly: Creative and cognitive aspects of implicitness in abusive language online, Natural Language Engineering 29 (2023) 1516–1537.*
- intensity
- offensiveness
- aggressiveness
- irony
- sarcasm


### Agents and Patients 
- *agent*: targets that are the subject of active verbs
- *patient*: targets that are object of the sentence or the subject of a passive verb 

 we performed a manual aggregation of Roma and Sinti in a unique category, as well as politicians including specific people and parties, and ethnic minorities named by referring to their origin, or with generic terms such as *foreigners*.


## 📜 Citation


