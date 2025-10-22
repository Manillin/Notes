# lezione 26 settembe  

power wall (alr talked about) and the memory wall (still witnessing) - speeds of the main memory is much slower than the speed performed by the cpu.  

this gives the birth of the multicore approach.  


designing simpler and slower cpus halves the power consumption and cuts the performance by a factor of 13% -> this is why we'll use multiple cores; based on the same example if we use two cores we get an increase of 73% on performance and a 2% power increase -> optimal tradeoff becasue we obtain a lot of performance gain for only a small increase in power consumption.   


dark silicon is space in the SoC where extra cpus would be cold because of use, we take that space and do something different that placing extra cpus.  
For example, taking dark silicon space and placing accelerators to better perform specific taks -> much more efficient in terms of energy and time.  


--- 

multiple processor systems advantages:  
- effetive use of billions of transistors 
- very easy scaling -> just add more cores 
- each single processor can be less powerful 

disadvantages: 
- parallelization is not a limitless way to increase performance, some programs are not naturally parallelized.  
- algorithms and hardware limit (interconnection of all the cores) the performance
- one task many processors - need to think how to share the task among multiple processors, and we need to carefully coordinate the processors.
- we need a new way of writing our programs to exploit the parallelism  




## Taxonomy of parallel computers:  

SISD: single instruction single data  
MISD: multiple instruction single data stream - not often used
SIMD: single instruction multiple data - one instruction being processed on multiple processing units - you process in the same cycle several operations (gpus are very large SIMD units)  
MIMD: multiple instruction and multiple data


vectorial units operate on a vector of elements (array in loop example)  
loop unrolling -> unroll the loop by a factor that matches the length of the vectorial unit.  



--- 

Shared Memory -> every core knows the address of the memory, only one copy of the data in main memory.  

- SMP
- DSM: Distributed shared memory, we have a Non-uniform access time (NUMA) meaning the processors closer to the actual cell of the data on the memory will access it faster than a processor which is further away.  


shared memory: novel issues 
1. coherence 
2. memory consistency 
3. synchronization (on writes! maybe a process is writing an item another process has in memory, making said process use old data and not the new changed data)  

p40 distributed memory but not shared, data is replicated in every processor's memory.  
in distributed memory you need to make the processors communicate via explicit message passing.  



--- 

coherence regards single location of data while consistencty handles the data to multiple locations .  
caches:
- Write through: when we update data it gets updated from l1cache to dram (not used a lot). 
Write back: uses a lazy approach, we publish the data from cache to the dram only when another process need to use the data that is different from cache of proc1 to the dram.   



---  


sections allows small form of task parallelism explotation. 

must highlight in the code the section of code that belongs to the sections part.   
It adds barrier by default but can be omitted w special clause.  

then you must annotate each individual task (sintassi).  

... 


sections can be used for other examples , and has a small overhead.  
under the hood it invoates a call to the run time and the run times does all the magic.  

what is the limit? -> all tasks have to be statically marked and some cases this is not feasible.  

what if a functional loop body is identified as a task ?  
a loop is a compact form of computation and doesn't match the requirements of the sections directive, one must unroll the loop to do so, and mark each iteration with the directive .   
but for large iterations this is not feasible.  
-> same goes for recursive functions.  

void traverse_list (List l)
{
Element e ;
#pragma omp parallel private ( e )
for ( l = l→first; l; l = l→next )
#pragma omp single nowait
process ( l ) ;
}

this function is traversing a list, l is the list variable.  


the pragma omp parallel clause prepares the threads, in the example above i would just do computation redundantly, each thread would do the whole loop.  


the clause to avoid all threads do the computation is: pragma omp single nowait  
with this directive the first thread that reaches the point does the computation, the other threads don't.  
with the nowait we are instruction the system to drop the barrier, the other threads skip that iteration and go to the next iteration (if they are the first thread to reach that iteration they will process that iteration and when the other threads when they reach that point will skip it; and so on and so forth).  [omp single implies mutually exclusive execution meaning we have a lock to synch, which means extra traffic for coherance].  
  

ricorsivamente è peggio, ad ogni chiamata ricorsiva abbiamo l'overhead di creazione , uccide la prestazione.  



we need better solution for problems of this nature.   

this was the main addition to openMP 3.0, allows to parallelize irregular problems.   

#pragma omp parallel:
- private(var_list) -> var list will be duplicated on each thread
- shared(var_list) -> var list will be shared, threads will share the same vars  

dead lock -> th1 is waiting for th2 but th2 is waiting for th1 , we are stalled and make no progress.   

task migration -> you start exe in a th but then the task migrates to another thread  
useful is load balancing (we want to make sure each thread is doing the same amount of work, if one is doing too much and one is doing none we could migrate)  

es:
T0 starts task1 and has a miss, we deschedule and add it to a queue, T0 starts tasks2, while task2 is executing we get the missed value from memory.  
we can schedule that to another free thread via migration!  


what is an openMP task:
- work units which execution may be deferred or executed immediately 

task directive: **#pragrma omp task [clauses]**
- each thread encountering a task will pack the code and data environment 

Highly composable and can be nested


on firstprivate (scoping) the values are captured at creation, so when we're migration we're passing also the stack of the task, it minimizes the traffic in the system.  
