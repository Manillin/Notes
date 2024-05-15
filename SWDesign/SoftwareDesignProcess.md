# Software Desing Process:

The quality of software is dictated by it's properties, specifically by

### Functional Properties:

We assess the **quality** of the software on Functional Properties such as:

- **Correctness $\rightarrow$** does my sw do what it needs to do (captured by the functional requirements which are discussed with the client)
- **Ease of Use $\rightarrow$** Includes the UX design but also readibility and usability of documentation
- **Performance/Efficiency $\rightarrow$** is it fast? efficient?
- **Dependability $\rightarrow$** Measures the reliability of the SW

### Non functional Properties:

Refers to the process of devolopment of the SW that could be unique to a specific domain or company, often hard to map onto product pricing.

- **Verifiability $\rightarrow$** Can we assess its characteristics
- **Maintainability $\rightarrow$** Is the SW easily modifiable
- **Reusability $\rightarrow$** Is it portable and well packed for deployment (docker)
- **Interoperability $\rightarrow$** Is the SW able to interact with other systems

<br><br>

# Basic Desgin Principles:

**<u>Strinctness and Formalism</u>** are key elements in designing quality SW

## 1. Separatio of Concern:

- **Divide et Impera $\rightarrow$** Splits the problem into subproblems.  
  There are many different subproblems in SW creation that need to be trated individually, the main ones are:

  - **Lifecycle $\rightarrow$ (Waterfall, SCRUM)**
  - **System Architecture $\rightarrow$ (Microservices)**
  - **Internal System Architecture $\rightarrow$ (MVC-MVVM)**
  - **Testing and Deployment $\rightarrow$ (CI/CD)**

- **Modularity $\rightarrow$** Affects the system design process and has two main approaches.

  - **Top-Down$\rightarrow$** Possible and recommended when we have a complete view of the project and split it into components $\rightarrow$ typically happens if we start from zero
  - **Bottom-Up$\rightarrow$** We first develop the singular components and then integrate them $\rightarrow$ happens if we're working on existing modules.

- **Abstraction $\rightarrow$** Must use the correct layer of abstraction depending on the problem at hand

## Main Issues with SW today (generational debt):
