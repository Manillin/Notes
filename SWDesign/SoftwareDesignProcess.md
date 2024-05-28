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

<br>

## Modern Issues with SW
The modern issues start with it's **cost**, SW costs outperform all other structural costs i.e. licences, fees for platforms, maintenance, personnel cost, ecc...

Other main problems are
- Generational Debt
- Increase in difficulty $\rightarrow$ complex and heterogeneous systems 
- Rushed Time to Market 


<br>


# Modelling the process

There are obviously many possibile paths since it hardly depens on:
- the type of tech
- the time at disposal for production (agile or more traditional approach)
- the legacy codebase (migrating or re-implementing), 
- the company process and many other variables.


<br>


### Typical Pattern:

1. **Requirements and Specifications (req and specs):**  
  Involve the client and _flood_ them with questions on the functional and non functional aspects; the environment;  the legacy codebase  
  The result from this step should be a document that coversa all relevant topics  $\rightarrow$ This is what defines our value for the customer, this is the **most important** phase , condsider it as making a promise to a customer, it's **key** to clearly state what we will do and what we won't do, and the responsabilites (who is in charge of the server infrastructure etc...)    
2. **Design:**  
  In this step we sketch a working system, and we make some imporant decisions like:
    - Architectural design (cloud, on premise, ...)
    - Choosing the language, frameworks, Hardware
    - Specify the team tools -> git, Agile, Waterfall
  We can strart drawing how our system is made and what tech it uses.  
    
3. **Implementation:**  
  If the design phase is well structured, the implementation is streamlined, make treasure of already established tech, no need to reinvent the wheel -> most apps follow CRUD, means there is a lot of frameworks to tackle that problem!  

4. **Integration:**  
  Consists in the process of reuniting the different parts of our software, ofc depends on the project (could be modules, components, services, ...)

5. **Testing & Valdiation:**  
  In this step we verify that we meet the specifications and requirements, and it's based in the concept of **Test Cases** at different abstaction levels.  
  Es:
    - store and modify info on user accounts -> high level of abs
    - store data only in a specific range -> lowe level, more details are given
    - if feb30th is inserted must return error -> lower level, specifies the constrain on a number (very specific test)

  There are different **Levels of Testing (in the large or small):**  
    1. **Unit testing:**  
      Tests a single component in isolation (es: a class)  
    2. **Module testing:**   
      Testing together more components that constitute a _functional block_  
    3. **Integration Testing:**   
      Tests that different components work together, like testing if a webserice interacts correctly with the DB.    

  A popular  testing structure consists in code coverage, using a white or black box approach, usually done by developers that didn't produce the code being tested.  

  When doing and designing tests is important to involve the customer to make the most out of it and avoid any misinterpretation.  
  
6. **Deploy:** 
  Most important aspect of deployment in today's age is the Continuous Integration / Continuos Deployment $\rightarrow$ **CI/CD**.  
  Often integrated in the testig phase, done by a framework and even better if automated.  

7. **Maintenance/AfterMarket**:
  The process of maintaining the SW free of bugs, for a period of time this process is included for free, later becomes a paid service (period of validity specified in the Reqs and Specs docs from step 1).  


