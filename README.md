# Julia for Systems Biology: Generating Personalized Models
JuliaCon 2024 Workshop

**[Shauna O'Donovan](https://research.tue.nl/en/persons/shauna-odonovan), [Max de Rooij](https://research.tue.nl/en/persons/max-de-rooij), [Natal van Riel](https://research.tue.nl/en/persons/natal-aw-van-riel)**

**Computational models are a valuable tool to study dynamic interactions and the evolution of systems behavior. Our hands-on and interactive workshop will demonstrate how personalized models can be more rapidly generated in Julia using various SciML packages combined with custom implementations. We will cover the implementation of ODE models in Julia, parameter estimation and model selection strategies including parameter sensitivity and identifiability analysis.**

Computational models offer a valuable tool for understanding the dynamic interactions between different biological entities, especially in biomedical applications. Personalizing these models with data can shed light on interindividual variation and project future health risks. However, model generation can be computationally expensive. Our hands-on and interactive workshop will demonstrate how personalized models can be more rapidly generated in Julia. We will be mainly using DifferentialEquations.jl combined with Optimization.jl and custom implementations of sensitivity and identifiability analysis approaches. Useing an in-house model of the glucose-insulin system, we will cover the implementation and resolving of ODE systems in Julia, including importing in SBML. We will provide a guide on model selection including parameter sensitivity and identifiability analysis, highlighting efficiencies that can be achieved using Julia. Additionally, we will discuss strategies for parameter estimation, including the benefits of regularization, using a publicly available data set of meal responses. Short presentation will be used to provide necessary background and theory and all methods will be implemented in a Jupyter notebook to facilitate independent learning.

## Contents
<!---
TODO: Add contents
-->
During the workshop, we will address the following elements of dynamic modelling in (systems) biology:
1. Implementation and simulation of (biological) dynamic models using `DifferentialEquations.jl`
2. Sensitivity analysis
3. Parameter identifiability analysis with profile likelihood
4. ...

## Program
The workshop starts with a short introductory presentation, outlining the general idea of systems biology and the explaining the goals of the workshop. The main part of the workshop will be split into three 45-minute hands-on sessions, where participants will implement and work with the [Eindhoven Diabetes Education Simulator (EDES)](https://pubmed.ncbi.nlm.nih.gov/25526760/) to learn about the different aspects of systems biology and dynamic modelling. 

### Date and Time
07-09, 09:00–12:00 (Europe/Amsterdam)

### Schedule
| Time | Description |
| ---- | ----------- |
| 09:00 | Introduction |
| 09:10 - 09:45 | Implementing the EDES Model, Simulation and visualisation |
| 09:45 - 10:00 | Short Break |
| 10:00 - 10:45 | Parameter Estimation, Personalisation |
| 10:45 - 11:00 | Short Break |
| 11:00 - 11:30 | Identifiability and profile likelihood |
| 11:30 - 11:45 | Hybrid Models, SciML |
| 11:45 - 12:00 | Closing, Evaluation and Discussion |

## The EDES Model
During the workshop, you will be implementing and working with the Eindhoven Diabetes Education Simulator (EDES). The EDES model is a mechanistic model of the glucose-insulin system, which is used to simulate the glucose-insulin dynamics in response to a meal. The model is originally developed in MATLAB by [Maas et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25526760/) and subsequently tranlated to Julia. The model is also available [in SBML format](https://www.ebi.ac.uk/biomodels/MODEL2403070001) in the BioModels database. A more detailed explanation of the model can be found [here](1_implementation/about_edes.md).

## Additional Information
* [SBML of the EDES Model](https://www.ebi.ac.uk/biomodels/MODEL2403070001)
