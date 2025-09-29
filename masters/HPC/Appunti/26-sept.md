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

