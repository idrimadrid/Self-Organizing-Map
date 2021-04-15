#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DATA_SIZE 150
#define VARIATION 0.05
#define VOISINS 2

//configuration network
struct N_conf
{
	int input_vector;	//data vector size
	int map_row;	// neuron map row number
	int map_col;	// neuron map column number
	int nb_iter;	// iteration number
	double alpha;	//starting alpha learning rate
}net_conf;

//initialisation du réseau
void init_net_conf()
{
	net_conf.input_vector = 4;
	net_conf.map_row = 10;
	net_conf.map_col = 6;
	net_conf.nb_iter = 3000;
	net_conf.alpha = 0.7;
}

//neuron  structure
typedef struct neuron neuron;
struct neuron
{
	double act_dist;	//euclidean distance
	char *label;
	double *weight;	//memory vector
};

//list BMU
typedef struct Bmu Bmu;
struct Bmu {
	double act; // euclidian distance
	int row;
	int col;
};

Bmu *bmu = NULL;
int bmu_size=1;

// réseau
typedef struct Net Net;
struct Net
{
	double *capteur; // vecteur selectionné
	neuron **map;	// carte
	int ray_vois;	// rayon voisinage
	double cur_alpha; // current alpha value
} net;

// Structure des données
typedef struct Data Data;
struct Data
{
	double *vec;
	double norm;
	char *label;
};

//Tableau qui contient structure des données
Data *data_tab;
// allocation tableau
void alloc_data_tab(int size)
{
	data_tab = malloc(sizeof(Data) *DATA_SIZE);
	int i;
	for (i = 0; i < size; i++)
	{
		data_tab[i].vec = malloc(net_conf.input_vector* sizeof(double));
		data_tab[i].norm=0;
		data_tab[i].label = malloc(sizeof(char));
	}
}
//vecteur moyen
double *average_vector;
// Mean Vector calc
void calc_average_vector()
{
    average_vector=calloc(net_conf.input_vector, sizeof(double));
    int i,j;

    for(i=0;i<net_conf.input_vector;i++)
    {
        for(j=0;j<DATA_SIZE;j++)
            average_vector[i]+=data_tab[j].vec[i];

        average_vector[i]/=DATA_SIZE;
    }
}

// randomize between 0 and 150
int random ()
{
	return (0 + (rand () % (DATA_SIZE-1)));
}
// gen valeur de variation de poids init
float variation()
{
    return (((float)rand()/(float)(RAND_MAX)) * VARIATION);
}
// gen neurons weights
double* init_neuron_weights()
{
    int i;
    double *w = malloc(net_conf.input_vector*sizeof(double));

    for(i=0;i<net_conf.input_vector;i++)
        {
        	if (random()%2==0)
            	w[i]=average_vector[i]+variation();
        	else
        		w[i]=average_vector[i]-variation();
        }
    return w;
    }

// map's neurons initialisation
void init_neuron_map()
{
    int i,j;
    net.map=malloc(net_conf.map_col*sizeof(neuron*));
	for(i=0;i<net_conf.map_col;i++)
	{
		net.map[i]=malloc(net_conf.map_row*sizeof(neuron));
	}
	for(i=0;i<net_conf.map_col;i++)
	{
		for (j=0;j<net_conf.map_row;j++)
		{
            net.map[i][j].weight=malloc(sizeof(double)*net_conf.input_vector);
			net.map[i][j].weight=init_neuron_weights();
			net.map[i][j].label=malloc(sizeof(char));
			net.map[i][j].label="*";
		}
	}
}

// normalize vector
void normal_vec(int i)
{
	double sum = 0;
	int j;
	for (j = 0; j < net_conf.input_vector; j++)
		sum += pow(data_tab[i].vec[j],2);
	data_tab[i].norm = sqrt(sum);
}

// Distance euclidienne
double distance(double *x, double *y)
{
    double somme = 0;
    for (int i = 0; i < net_conf.input_vector; i++)
    {
        somme += pow(x[i] - y[i], 2);
    }
    return sqrt(somme);
}

//randomise les données
void shuffle( Data *tab)
{
	int index;
	Data temp;
	int i;
	for (i=0; i<DATA_SIZE; i++)
	{
        index = random();
		temp = tab[i];
		tab[i] = tab[index];
		tab[index] = temp;
	}
}

// chargement des données
void load_data()
{
	FILE * input;
	input = fopen("iris.data", "r");
	char str[50];
	int i, j;
    for(i=0;i<150;i++)
	{
	    fscanf(input,"%s",str);
		char *sep = strtok(str, ",");
		for (j = 0; j < net_conf.input_vector; j++)
		{
			data_tab[i].vec[j] = atof(sep);
			sep = strtok(NULL, ",");
		}
        if (strcmp(sep, "Iris-versicolor") == 0)
            data_tab[i].label="V";
		else if (strcmp(sep, "Iris-setosa") == 0)
			data_tab[i].label="C";
		else
			data_tab[i].label="I";

		normal_vec(i);

	}
	fclose(input);
}

// show loaded data
void show_data(Data *data)
{
	int i =0; int j= 0;
	for(i=0;i<DATA_SIZE; i++) {
            printf("data[%d] : ", i);
		for(j=0;j<4; j++) {
			printf("  %f  ", data[i].vec[j]);
		}
		printf("label: %s \n", data[i].label);
	}
}

// alpha coeff reduction
void update_alpha(int it_n)
{
	net.cur_alpha = net_conf.alpha * (1 - ((double)it_n/(double)net_conf.nb_iter));
}

//Update weights of the BMU and neighbors
void update(Bmu* _bmu)
{
    int voisin=net.ray_vois;
    int i,j,x1,x2,y1,y2;//


        if(_bmu->row-voisin<0)
            x1=0;
        else
            x1=_bmu->row-voisin;

        if(_bmu->row+voisin>net_conf.map_col-1)
            x2=net_conf.map_col-1;
        else
            x2=_bmu->row+voisin;

        if(_bmu->col-voisin<0)
            y1=0;
        else
            y1=_bmu->col-voisin;

        if(_bmu->col+voisin>net_conf.map_row-1)
            y2=net_conf.map_row-1;
        else
            y2=_bmu->col+voisin;

        for(i=x1;i<=x2;i++)
            for(j=y1;j<=y2;j++)
            {
                int k;
                for(k=0;k<net_conf.input_vector;k++)
                    {
                        //if(i==_bmu->row && j==_bmu->col)
                            net.map[i][j].weight[k]+=net.cur_alpha*(net.capteur[k]-net.map[i][j].weight[k]);

                        /*else if(i==_bmu->row+1 || j==_bmu->col+1 || i==_bmu->row-1 || j==_bmu->col-1)
                            net.map[i][j].weight[k]+=0.75*net.cur_alpha*(net.capteur[k]-net.map[i][j].weight[k]);

                        else
                            net.map[i][j].weight[k]+=0.5*net.cur_alpha*(net.capteur[k]-net.map[i][j].weight[k]);*/
                    }
            }

}

// display topology
void show_maps(int it)
{
    printf("Epoch n° : %d \n",it);
    printf("================================================\n");
    int i,j;
    for(i=0;i<net_conf.map_col;i++)
    {
        printf("\n");
        for(j=0;j<net_conf.map_row;j++)
            {
                printf(" %s ",net.map[i][j].label);
            }
            printf("\n");
        }
        printf("\n");
        printf("================================================\n");
}

// training
void train()
{
    int i,j,k,it;
    double min_dist,dist;
    bmu=malloc(sizeof(Bmu));
        for(it=1;it<=net_conf.nb_iter;it++)
        {
            for(k=0;k<DATA_SIZE;k++)
            {
                net.capteur=data_tab[k].vec;
                min_dist=100.0;
                for(i=0;i<net_conf.map_col;i++)
                {
                    for(j=0;j<net_conf.map_row;j++)
                    {
                        dist=distance(net.capteur,net.map[i][j].weight);
                        net.map[i][j].act_dist=dist;
                        if(dist<min_dist)
                        {
                            min_dist=dist;
                            if(bmu_size>1)
                            {
                                bmu_size=1;
                                bmu=realloc(bmu,bmu_size*sizeof(Bmu));
                            }
                            bmu[0].act=dist;
                            bmu[0].row=i;
                            bmu[0].col=j;
                        }
                        else if(dist==min_dist)
                        {
                            bmu_size++;
                            bmu=realloc(bmu,bmu_size*sizeof(Bmu));
                            bmu[bmu_size-1].act=dist;
                            bmu[bmu_size-1].row=i;
                            bmu[bmu_size-1].col=j;
                        }
                    }
                }
                //Choix aléatoire du BMU
                if(bmu_size>1)
                {
                    int t=rand()%(bmu_size);
                    bmu[0]=bmu[t];
                }

                net.map[bmu[0].row][bmu[0].col].label= data_tab[k].label;
                update(bmu);

                // réduction voisinage
                if(it%(net_conf.nb_iter/3)==0)
                    net.ray_vois-=1;


            }

            update_alpha(it);
            shuffle(data_tab);
        }
    }

//save in file
void save_weights()
{
    time_t current_time;
    current_time = time(NULL);
    int i,j;
    char* txt;
    asprintf(&txt,"%d_%d_X_%d_%d.txt",current_time,net_conf.map_row,net_conf.map_col,net_conf.nb_iter);

    FILE * file;
    file = fopen(txt, "w");
    char str[100];

    for(i=0;i<net_conf.map_col;i++)
	{
		for (j=0;j<net_conf.map_row;j++)
		{
		    sprintf (str, "%lf,%lf,%lf,%lf,%s\n",net.map[i][j].weight[0],net.map[i][j].weight[1],net.map[i][j].weight[2],net.map[i][j].weight[3] ,net.map[i][j].label);
		    fputs(str, file);
		}
	}
    fclose(file);
}

void load_weights(char *file)
{
    FILE * input;
	input = fopen(file, "r");
	char str[50];
	int i, j,k;

		for(i=0;i<net_conf.map_col;i++)
        {
            for (j=0;j<net_conf.map_row;j++)
            {
                fscanf(input,"%s",str);
                char *sub_string = strtok(str, ",");

                for (k = 0; k < net_conf.input_vector; k++)
                {
                    net.map[i][j].weight[k] = atof(sub_string);
                    sub_string = strtok(NULL, ",");
                }
                net.map[i][j].label=sub_string;
            }
        }
        fclose(input);
	}

int main()
{
    char ans;
    net.ray_vois=VOISINS;

	init_net_conf();
	alloc_data_tab(DATA_SIZE);
	load_data();
	shuffle(data_tab);
	//show_data(data_tab);

	calc_average_vector();
	init_neuron_map();
	train();

	//load_weights("1610727332_8_X_8_300.txt");
    show_maps(0);

    printf("Voulez-vous sauvegarder le modele ? o/n\n");
    scanf("%c",&ans);
    //if (ans=="o")
        save_weights();

	return 0;
}
