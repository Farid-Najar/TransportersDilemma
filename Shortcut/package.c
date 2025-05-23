#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

//the tours are given as a sequence of vertices
//vertex 0 is the warehouse 
//it cannot be removed from the tour

typedef struct{ //store the tour
	int length;
	int *vertices;
}tour;

typedef struct{ //store the graph
    int vertex_number;
    float **distance;
} graph;


typedef struct{ //store the table from dynamic programming
    int max_omitted;
    float **values; //values[k][i]: max gain, obtained by removing k element before vertex i
    int **sol; //store the number of elements skipped when computing the optimal value
} table;

tour init_tour(int n){
    tour t = {n,malloc(n*sizeof(int))};
    return t;
}

tour random_tour(int m, int n){
    tour t = init_tour(n);
    t.vertices[0] = 0;
    t.vertices[n-1] = 0;
    for(int i = 1; i < n-1; i++){
        t.vertices[i] = rand()%m; 
    }
    return t;
}

void initialize_and_permute(int *permutation, int n)
{
    int i;
    for (i = 0; i <= n-1; i++) {
        permutation[i] = i;
    }

    for (i = 0; i < n-1; i++) {
        int j = i + rand() %(n-i); /* A random integer such that i ≤ j < n */
        int temp = permutation[j];
        permutation[j] = permutation[i]; /* Swap the randomly picked element with permutation[i] */
        permutation[i] = temp;
    }
}

tour random_perm(int n){
    tour t = init_tour(n);
    initialize_and_permute(t.vertices,n);
    t.vertices[0] = 0;
    t.vertices[n-1] = 0;
    return t;
}



void show_tour(tour t){
    printf("Tour avec %d étapes.",t.length);
    for(int i = 0; i < t.length; i++){
        printf(" %d",t.vertices[i]);
    }
    printf("\n");
}

graph init_graph(int n){
    graph g = {n,malloc(n*sizeof(float*))};
    for(int i = 0; i<n; i++){
        g.distance[i] = calloc(n,sizeof(float));
    }
    return g;
}

graph random_graph(int n){
    graph g = init_graph(n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < i; j++){
                g.distance[i][j] = 1 + (float)rand()/(float)(RAND_MAX);
                g.distance[j][i] = g.distance[i][j];
        }
        g.distance[i][i] = 0;
    }
    return g;
}


void show_graph(graph g){
    printf("\n Graph with %d vertices \n",g.vertex_number);
    for(int i = 0; i < g.vertex_number; i++){
        printf("Vertex %d: ",i);
        for(int j = 0; j < g.vertex_number; j++){
                printf("%d: %f, ", j,g.distance[i][j]);
        }
        printf("\n");
    }
}


table init_table(int tour_length, int K){//K is the largest number of packages to remove
    table t;
    t.max_omitted = 0;
    t.sol = malloc(K*sizeof(int*));
    t.values = malloc(K*sizeof(float*));
    t.values[0]= calloc(tour_length,sizeof(float));
    return t;
}

int *get_solution_single_type(table tab, int k, int tour_length){//get an optimal solution with k omitted packages from a full dynamic table
    int *sol = malloc(k*sizeof(int));
    int position = tour_length-1;
    //printf("\n K max considéré: %d Position max : %d\n",k,position);
    while(k){//continue until it has found all packets to remove
        int to_remove = tab.sol[k][position];
        //printf("%d %d %d\n",position,k,to_remove);
        for(int i = 1; i <= to_remove; i++ ){
            k--;
            sol[k] = position -i;
        }
        position -= to_remove+1;
    }
    return sol;
}


int **get_solution_multiple_types(table *tab, int *max_sol, tour *tour, int types){//get a solution from a full dynamic table
    int **sol = malloc(types*sizeof(int*));
    for(int i = 0; i< types; i++){
        sol[i] = get_solution_single_type(tab[i],max_sol[i],tour[i].length);
    }
    return sol;
}

float **compute_delta(graph g, tour t, int k){//compute the difference in cost when removing elements in the tour
    float** delta = malloc(t.length*sizeof(float*));
    //no need for delta[0]
    for(int i = 1; i < t.length; i++){ 
        delta[i] = calloc(k+1,sizeof(float));
        int current_vertex = t.vertices[i]; 
        float sum = g.distance[current_vertex][t.vertices[i-1]];
        //printf("\n Removing before vertex %d :",i);
        for(int j = 1; j <= k; j++){
            if(t.vertices[i-j]){//the vertex is not 0, hence can be removed
                sum += g.distance[t.vertices[i-j-1]][t.vertices[i-j]];
                delta[i][j] = sum - g.distance[t.vertices[i-j-1]][current_vertex];
                //printf("(%d,%f)  ", j ,delta[i][j]);
            }
            else{//the vertex is 0, we cannot remove
                delta[i][j] = -1; //special value to escape from the computation
                break;
            }
        }
    }
    return delta;
}

table compute_smallest_cost(graph g, tour t, float excess, int K){ 
    float **delta = compute_delta(g,t,K+1);
    table tab = init_table(t.length,K+1);
    for(int k = 1; tab.values[k-1][t.length-1] < excess && k <= K; k++){//while the pollution constraint is violated, try to remove one additional package
        tab.max_omitted = k;
        tab.values[k] = calloc(t.length,sizeof(float));
        tab.sol[k] = calloc(t.length,sizeof(int));
        for(int i = k+1; i < t.length; i++){//we may begin from k+1, because we need to remove k elements before it and vertex 0 cannot be removed
            //loop to determine the optimal number of elements to remove before i
            for(int j = 0; delta[i][j] != -1 && j <= k; j++){//break when an element cannot be removed
                float val = delta[i][j] + tab.values[k-j][i-j-1];
                if(val > tab.values[k][i]){
                    tab.values[k][i] = val;
                    tab.sol[k][i] = j;
                    if(j>1) printf("Trouvé %d",j);
                }
            }
        }
    }
    // //print for debugging
    // for(int i = 0; i <= tab.max_omitted; i++){
    //     printf("\n Number of ommited packages %d, gain %f",i,tab.values[i][t.length -1]);
    // }
    // int *s = get_solution(tab,tab.max_omited,t.length);
    // printf("Position of packages omitted in the solution: ");
    // for(int i = 0; i < tab.max_omitted; i++){
    //     printf("%d ",s[i]);
    // }
    return tab;
}


float value(float **value_tables, float* coeff, int types, int * sol, float excess){//evaluate the cost and pollution reduction of a solution, return the cost reduction (gain) if the pollution 
                                                                                    //reduction is sufficient (larger than excess)
    float gain = 0;
    float pol = 0;
    for(int i = 0; i < types; i++){
        gain += value_tables[i][sol[i]];
        pol += coeff[i]*value_tables[i][sol[i]];
    }
    return pol > excess ? gain : 0; 
}

void best_combination(int k, int types, int current_type, float **value_tables, float *coeff, float excess, int *sol, float *max_val, int* max_sol){//extremly simplistic enumeration of the way to generate k
                                                                                                                                                  //as a sum of t positive integers
    int total = 0;
    for(int i = 0; i < current_type; i++){
        total+= sol[i];
    }
    //printf("Total %d current type %d\n",total,current_type);
    if(current_type == types -1){//last type, we know the value we have to choose
        
        sol[current_type] = k - total;
        float val = value(value_tables,coeff,types,sol,excess);
        //printf("value : %f \n",val);
        if (val > *max_val){
            *max_val = val;
            memcpy(max_sol,sol,types*sizeof(int));
        }
        return; //a solution is found, stop
    }
    
    for(int i = 0; i <= k - total; i++){
        sol[current_type]=  i;
        best_combination(k, types, current_type+1, value_tables, coeff, excess, sol, max_val, max_sol);
    }
}

void multi_types(graph g, tour *t, float *coeff, int types, float excess){//types is the number of types (and thus of tour and coeff)
                                                                                 //it is better for perf to give as first tour the one with the largest coeff (more polluting)
    table *tab = malloc(types*sizeof(table)); //one table for each tour
    tab[0] = compute_smallest_cost(g, t[0], excess/coeff[0], t[0].length);
    int max_omission = tab->max_omitted;
    for(int i = 1; i < types; i++){
        tab[i] = compute_smallest_cost(g, t[i], excess/coeff[i], max_omission);
        if(max_omission > tab[i].max_omitted) max_omission = tab[i].max_omitted;
    }//compute the optimal value, depending on the number of omission, for each tour
    //extract the best combination of omission between the different types
    float **value_tables = malloc(types*sizeof(float*));
    for(int i = 0; i < types; i++){
        value_tables[i] = malloc((max_omission+1)*sizeof(float));
        for(int j = 0; j <= max_omission; j++){
            value_tables[i][j] = tab[i].values[j][t[i].length -1];
        }
    }
    
    int *sol = calloc(types, sizeof(int));
    int *max_sol = calloc(types, sizeof(int));
    float *max_val = calloc(1,sizeof(float));
    for(int k = 1; k < max_omission+1; k++){//we could compute the best improvement in pollution by removing a single element (or even by removing l elements) -> lower_bound of this loop + cut in the recursive algorithm
        //printf("k: %d \n",k);
        best_combination(k, types, 0, value_tables, coeff, excess, sol, max_val, max_sol);
        if(*max_val != 0){
            break;
        }
    }
    //Code to print the solution
    int **final_sol = get_solution_multiple_types(tab,max_sol,t,types);
    printf("best solution of value: %f\n",*max_val);
    for(int i=0; i < types;i++){
        printf("%d packages omitted in tour %d: ",max_sol[i],i);
        for(int j = 0; j < max_sol[i];j++){
            printf("%d, ",final_sol[i][j]);
        }
        printf("\n");
    }
}



int main(){

srand(2);  

//simple example to test the algorithm
// tour t = init_tour(4);
// t.vertices[0]=0;
// t.vertices[1]=1;
// t.vertices[2]=2;
// t.vertices[3]=0;
// graph g = init_graph(3);
// for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 3; j++){
//             g.distance[i][j] = i+j%2;
//     }
// }
// //call the code to test on this simple example, should maximize the value
// compute_smallest_cost(g, t, 1 , 1);

// //larger random example
// graph g = random_graph(100);
// //show_graph(g);
// tour t = random_perm(100);
// //show_tour(t);
// compute_smallest_cost(g, t, 30, 20);


//example with three types of trucks
graph g = random_graph(100);
float *coeff = malloc(3*sizeof(float));
coeff[0]= 0.8;
coeff[1]= 0.9;
coeff[2]= 1;
tour *tht = malloc(3*sizeof(table));
tht[0] = random_perm(100);
tht[1] = random_perm(100);
tht[2] = random_perm(100);
multi_types(g, tht, coeff, 3, 50);
}