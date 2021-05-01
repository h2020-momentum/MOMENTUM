
## Assign destination choice
 
Inputs: 
* Aggregate OD matrices (could be provided per mode, trip purpose).
* Synthetic population generated using the PopGen tool. 
* Zoning system information (census tract, TAZ)
* Steet map

### Origin-Destination trips ###

<p> 1. Calculate the total number of trips from each origin zone i (i=1,…,N): T

<p> 2. For each origin zone i calculate the number of trips to each destination zone j (j=1,…,N): T<sub>(i,j)</sub>

<p> 3. Derive the probability of a trip from each origin zone i to a particular destination zone j: P<sub>(i,j)</sub> = T<sub>(i,j)</sub>/T<sub>i</sub>    

<p> 4. Assign destinations to the synthetic population:

#### Approach 1: Random destination assignment ####

Based on P<sub>(i,j)</sub> assign randomly a destination to each person in the synthetic population for each home zone i.  

#### Approach 2: Impute exact origin and destination locations ####

Depending on the required spatial resolution, specific origin and destination locations at the level of x,y coordinates, could be also assigned:

<p> 1. Count the number of individuals in the synthetic population that have their origin (home) inside each zone i. 

  * Let S be list with synthesised population

  * Let O<sub>i</sub>[] be the demand total for each zone  i 

<pre>
 <b>for each i in S do</b>
  O<sub>i</sub>+=1 
 <b>end for </b>
</pre>

<p> 2. For each origin zone i sample trip counts to the destination zones i from a multinomial distribution, given the likelihood P<sub>(i,j)</sub> that a trip between an origin i and destination j exists:
<pre>
<b>for each i do</b>
 sample(C<sub>(i,1)</sub>,…,C<sub>(i,N)</sub>) = Multinomial(O<sub>i</sub>,P<sub>(i,j)</sub>)
<b>end for </b>
∑<sub>j</sub>C<sub>(i,j)</sub> = O<sub>i</sub>
</pre>

<p> 3. For a combination of zones i,j sample coordinates for C<sub>(i,j)</sub>  destinations randomly.

<p> 4. Define a set of coordinates destinations can be assigned to the synthetic population. 
	
<p> 5. For each zone i, we can now determine all persons n with an origin (home) located in that zone, n∈{1,…,O<sub>i</sub> }
	
<p> 6. The candidate coordinates in each zone i  are c∈{1,…,O<sub>i</sub> }

<p> 7. Randomly assign a destination to each person in the synthetic population within each zone i

<pre>
  <b>for each n in i  do</b>
   c=A(n)=n
  <b>end for </b>
</pre>
