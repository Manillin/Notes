# Entity Relationship Model (ER)  

Is a conceptual model for designing DBs, represents the logical structure, including entities their attributes and the relationships between them.  

1. entity: ojbect that is stored as data (es: student, course, ...) 
2. Attribute: properties that describe an entity (es: studentID, CourseName, ...) 
3. relationships: connection between entities (es: student enrolles in a course)  



Cardinality means what is the number of relationships between the two entity sets in any relationship model. There are four types of cardinality which are mentioned below:

1. One-to-One (1:1): Each entity in set A is related to at most one entity in set B and vice versa.
2. One-to-Many (1:N): An entity in set A can be related to many entities in set B, but each entity in B is related to only one entity in A.
3. Many-to-One (N:1): Opposite of One-to-Many.
4. Many-to-Many (M:N): Entities in both sets A and B can be related to multiple entities in the other set  



Types of keys:
1. Primary Key: uniquely identifies each tuple(= row in a relation) in a relation, it must contain unique values and cannot have NULL values.  
2. Candidate Key: set of attributes that can uniquely identify a tuple in a relation
3. Super key: set of attributes that can identify a tuple uniquely 
4. Foreign Key: attribute in one relation that refers to the primary key of another relation
5. Composite Key: is formed by combining two or more attributes to uniquely identify a tuple 

--- 

Ogni entità deve possedere almeno un identificatore (ma potrebbe averne più di uno)  

Una IDENTIFICAZIONE ESTERNA è possibile solo attraverso una relationship a cui l'identità da identificare partecipa con cardinalità (1,1)  

