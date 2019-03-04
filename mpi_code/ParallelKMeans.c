#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <argp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

const double DEF_TRESHOLD = 0.00001;
const int DEF_CENTROIDS = 2;
const int MAX_ITER = 50;
const int FILE_NOT_FOUND = 1;
const int ROOT_PROCESS = 0;
const int MATRIX_DIM = 2;
const int MATRIX_ROW = 0;
const int MATRIX_COLUMN = 1;
const int M_KEY = 0;
const int M_VALUE = 1;
const int MAP_SIZE = 2;

//utilizzo questo valore [specificato da linea di comando], se viene settato a uno
//non stampo nulla e misuro i tempi d'esecuzione, altrimenti i risultati elaborati dal programma
//verranno presentati in output;
int benchmark;

struct mpi_state_container {
	int nproc;
	int rank;
	int err;
};

struct mk_conf {
	double threshold;
	char *filepath;
	int centroids;
	int max_iterations;
	int err;
};

struct matrix_size {
	int rowlen;
	int collen;
	int err;
};

struct file_reader {
	int array_size;
	int start_line;
	int rows_to_read;
};

struct line_fetcher {
	double number;
	int err;
};

struct matrix_container {
	double *data;
	int *size;
};

struct cntr_trckr {
	struct matrix_container *nwcntr;
	int *total;
};

struct chunk_container {
	struct matrix_container *centroids;
	struct matrix_container *points;
	struct matrix_container *map;
	int offset;
};

//analizzo parametri che vengono passati da linea di comando;
static int parse_arg(int key, char *arg, struct argp_state *state) {
	struct mk_conf *config = state->input;
	switch(key) {
	case 't': {
		config->threshold = atof(arg);
		break;
	}
	case 'f': {
		config->filepath = arg;
		break;
	}
	case 'c': {
		config->centroids = atoi(arg);
		break;
	}
	case 'i': {
		config->max_iterations = atoi(arg);
		break;
	}
	case 'b': {
		benchmark = 1;
		break;
	}
	}

	return 0;
}

//setto i parametri accettati da linea di comando, nel caso in cui non vengano specificati, vengono assegnati,
//ove possibile, dei valori di default
struct mk_conf extract_parameters(int n, char **v) {
	struct mk_conf config;
	config.threshold = DEF_TRESHOLD;
	config.centroids = DEF_CENTROIDS;
	config.filepath = NULL;
	config.max_iterations = MAX_ITER;
	config.err = 0;
	benchmark = 0;

	struct argp_option options[] = {
		{"treshold",'t', "FLOAT", 0, "Insert the threshold"},
		{"fpath",'f', "STRING", 0, "insert the file path"},
		{"centroids",'c', "INT", 0, "Insert the number of centroids"},
		{"maxiter",'i', "INT", 0, "Insert maximum number of iterations"},
		{"benchmark",'b', "BOOL", 0, "if benchmark nothing will be written in files"},
		{0}
	};

	struct argp arg_opt = { options, parse_arg, 0, 0 };

	config.err = argp_parse(&arg_opt, n, v, 0, 0, &config);

	return config;
}

int char_cmp(char a, char b) {
	return (a == b) ? 1 : 0;
}

//conto il numero di sep contenute nella linea in input [serve per ricavare il numero di colonne]
int count_sep(char *line, char sep) {
	int occ = 0;
	while((*line) && (*line != '\n')) {
		occ += char_cmp(*(line++), sep);
	}
	return occ;
}

struct matrix_size init_mz(int rows, int cols) {
	struct matrix_size mz;
	mz.collen = cols;
	mz.rowlen = rows;
	mz.err = 0;
	return mz;
}

struct matrix_size rowcollen(FILE *f, char sep) {
	char* line = NULL;
	int rows = 0, cols = 0;
	size_t len = 0;
	ssize_t read;
	//ricavo le linee e le colonne contenute nel file;
	while((read = getline(&line, &len, f)) != -1) {
		if (rows == 0) {
			cols = count_sep(line, sep) + 1;
		}
		rows++;
	}
	if(line) {
		free(line);
	}
	return init_mz(rows, cols);
}


struct matrix_size init_lengths(char *fpath, char sep) {
	FILE *fp = fopen(fpath, "r");
	struct matrix_size mz = init_mz(0, 0);
	if (!fp) {
		mz.err = FILE_NOT_FOUND;
		return mz;
	}
	mz = rowcollen(fp, sep);
	fclose(fp);
	return mz;
}

struct mpi_state_container init_mpi_state(int *argc, char ***argv) {
	struct mpi_state_container msc;
	MPI_Init(argc, argv);
	MPI_Comm_size(MPI_COMM_WORLD, &(msc.nproc));
	MPI_Comm_rank(MPI_COMM_WORLD, &(msc.rank));
	return msc;
}

//visualizza un messaggio di errore e manda un abort a tutti i processi;
int abort_app(char *error_message, int err_code) {
	fprintf(stderr, "%s", error_message);
	MPI_Abort(MPI_COMM_WORLD, err_code);
	if (error_message) {
		free(error_message);
	}
	return err_code;
}

int *alloc_int(int size) {
	return malloc(size * sizeof(int));
}

//inizializzo l'array che mantiene le dimensioni delle matrici;
int *to_container(int row, int col) {
	int *size = alloc_int(MATRIX_DIM);
	size[MATRIX_ROW] = row;
	size[MATRIX_COLUMN] = col;
	return size;
}

int *m_size_to_array(struct matrix_size m_size) {
	return to_container(m_size.rowlen, m_size.collen);
}

int *dim_array(struct matrix_size m_size, int rank) {
	return (rank == ROOT_PROCESS) ? m_size_to_array(m_size) : alloc_int(MATRIX_DIM);
}

int* broadcast_matrix_size(struct matrix_size m_size, int rank) {
	int *size = dim_array(m_size, rank);
	//mando la dimensione del data frame a tutti i processi;
	MPI_Bcast(size, MATRIX_DIM, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
	return size;
}

int get_row(int *container) {
	return container[MATRIX_ROW];
}

int get_col(int *container) {
	return container[MATRIX_COLUMN];
}

double* alloc_double(int dim) {
	return malloc(dim * sizeof(double));
}

char* alloc_char(int dim) {
	return malloc(dim * sizeof(char));
}

int line_start(int *rc, struct mpi_state_container *state) {
	return (get_row(rc) / (*state).nproc)*(*state).rank;
}

int get_array_dimension(int *rc, int num_proc) {
	return (get_row(rc) * get_col(rc))/num_proc;
}

struct file_reader init_reader(int *rc, struct mpi_state_container *state) {
	struct file_reader readr;
	readr.start_line = line_start(rc, state);
	readr.rows_to_read = get_row(rc) / (*state).nproc;
	readr.array_size = get_array_dimension(rc, (*state).nproc);
	return readr;
}

int read_until(char *line, int from, char sep, double *res){
	int ta = 0; 

	for(ta = from; line[ta] != '\n' && line[ta] != sep; ta++);
	int alloc_i = ta - from;
	char *tc = alloc_char(alloc_i);
	for(int i = 0; i < alloc_i; i++){
		tc[i] = line[i+from];
	}	
	
	*res = atof(tc);

	free(tc);
	if (line[ta] == '\n'){
		return -1;
	}
	return ta + 1; 
}

int parse_line(char *line, char sep, double *local, int starting_point, int max) {
	char *duplicate = strdup(line);
	char *free_it = duplicate;
	char *token;
	int index = starting_point;
	int to_read = 0;
	int do_i = 0;
	while(to_read != -1){
		to_read = read_until(duplicate, to_read, sep, &(local[index]));
		//printf("%f\n", local[index]);
		index++;
	}
	if(free_it) {
		free(free_it);
	}
	return index - starting_point;
}

//inizializzo l'array locale;
double *init_local_array(struct file_reader *readr, char* filepath, char sep) {
	FILE *fp = fopen(filepath, "r");
	char *line = NULL;
	size_t len = 0;
	double *local_matrix = alloc_double((*readr).array_size);
	//alloco la matrice locale
	//calcolo il limite massimo delle linee da leggere
	int limit = (*readr).start_line + (*readr).rows_to_read;
	int written = 0;
	for (int i = 0; ((i < limit) && (getline(&line, &len, fp) != -1)) ; i++) {
		if (i >= (*readr).start_line) {
			//questa funzione estrae i valori dalle righe del file;
			written += parse_line(line, sep, local_matrix, written, readr->array_size);
		}
	}
	if(line) {
		free(line);
	}
	fclose(fp);
	if (written != (*readr).array_size) {
		printf("hey! found: %d elements; should be %d\n", written, (*readr).array_size);
	}
	return local_matrix;
}

struct matrix_container* alloc_matrix_container() {
	return malloc(sizeof (struct matrix_container));
}

//alloco la struttura matrix_container [i dati interni sono già stati calcolati]
struct matrix_container* init_matrix(double *matrix_cnt, int *row_col) {
	struct matrix_container *mc = alloc_matrix_container();
	mc->data = matrix_cnt;
	mc->size = row_col;
	return mc;
}

//alloco la struttura matrix_container [i dati non sono stati allocati]
struct matrix_container* alloc_matrix_data(int *row_col) {
	struct matrix_container *mc = alloc_matrix_container();
	mc->data = alloc_double(get_row(row_col) * get_col(row_col));
	mc->size = row_col;
	return mc;
}

//alloco una matrice
struct matrix_container* alloc_basic_matr(int rows, int cols) {
	int *d_size = alloc_int(MATRIX_DIM);
	d_size[MATRIX_ROW] = rows;
	d_size[MATRIX_COLUMN] = cols;
	return alloc_matrix_data(d_size);
}

//alloco i centroidi
struct matrix_container* alloc_centroids(int *original_size, int ncentroids) {
	return alloc_basic_matr(ncentroids, get_col(original_size));
}

//alloco la mappa
struct matrix_container* alloc_map(int *original_size) {
	return alloc_basic_matr(get_row(original_size), MAP_SIZE);
}

//non so sta funzione cosa ci faccia qui.
int compare(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}

//ricavo l'indice; r e c rappresentano rispettivamente la riga e la colonna richiesti in input;
int get_index(struct matrix_container *mc, int r, int c) {
	return r * get_col(mc->size) + c;
}

//prendo l'elemento dalla matrice;
double get_element_from(struct matrix_container *mc, int r, int c) {
	return mc->data[get_index(mc, r, c)];
}

//setto l'elemento nella matrice;
void set_element_in(struct matrix_container *mc, int r, int c, double val) {
	mc->data[get_index(mc, r, c)] = val;
}

//stampo la matrice
void print_m(struct matrix_container *mc) {
	int ncol = get_col(mc->size);
	int nrow = get_row(mc->size);

	for (int i = 0; i < nrow; i++) {
		for(int j = 0; j < ncol; j++) {
			printf("%f\t", get_element_from(mc,i,j));
		}
		printf("\n");
	}
	printf("\n\n");
}

void random_centroids(struct chunk_container *chunk) {
	int ncentr = get_row(chunk->centroids->size);
	int max_number = get_row(chunk->points->size);
	int ncol = get_col(chunk->centroids->size);
	srand(-time(NULL));
	//assegno punti random come centroidi iniziali, tramite knuth
	for(int i = 0, taken = 0; i < max_number && taken < ncentr; i++) {
		//tenfo traccia dei punti presi con taken;
		if(rand() % (max_number-i) < (ncentr-taken)) {
			//se il resto della divisione tra un un numero random e le righe restanti è minore
			//rispetto ai centroidi da prendere, il punto viene selezionato come centroide;
			for(int j = 0; j < ncol; j++) {
				//copio la linea nella matrice dei centroidi.
				set_element_in(chunk->centroids, taken, j, get_element_from(chunk->points, i, j));
			}
			++taken;
		}
	}
}

//inizializzo la struttura locale, mantiene una matrice per i punti del data frame, una per i centroidi e una per la mappa;
//in più tiene traccia dell'offset [viene utilizzato per assegnare un id ad ogni punto];
struct chunk_container* build_chunk_container(double *matrix_cnt, int start_line, int *row_col, struct mk_conf *params) {
	struct chunk_container *chunk = malloc(sizeof (struct chunk_container));
	chunk->points = init_matrix(matrix_cnt, row_col);
	chunk->centroids = alloc_centroids(row_col, params->centroids);
	chunk->offset = start_line;
	chunk->map = alloc_map(row_col);
	return chunk;
}

//calcolo la distanza euclidea tra un punto del dataframe ed il centroide
//la coordinata del punto viene specificata in p, mentre quella del centroide da ce;
double euclidean_distance(struct chunk_container *c, int p, int ce, int ndims) {
	double dist = 0.0;
	for (int d = 0; d < ndims; d++) {
		dist += pow(get_element_from(c->points, p, d) - get_element_from(c->centroids, ce, d), 2);
	}
	return sqrt(dist);
}

struct minimum_distance {
	double dist;
	int closest_point;
	int dirty;
};

//inizializzo md;
struct minimum_distance* init_md() {
	struct minimum_distance *md = malloc(sizeof(struct minimum_distance));
	md->dist = 0.0;
	md->closest_point = 0;
	md->dirty = 0;
	return md;
}

// assegna i valori a md
void set_md_values(struct minimum_distance *md, double d, int cp) {
	md->dist = d;
	md->closest_point = cp;
	md->dirty = 1;
}

//se il punto in input è il più vicino tra quelli considerati finora, allora lo manteniamo nella struttura di supporto.
//l'attributo dirty viene utilizzato per tenere traccia della prima volta che la struttura viene utilizzata, nel caso,
//l'elemento viene assegnato senza effettuare alcun tipo di controllo.
void update_md(struct minimum_distance *md, double d, int cp) {
	if ((md->dirty == 0) || ((md->dirty == 1) && (md->dist > d))) {
		set_md_values(md, d, cp);
	}
}

void keep_track(struct chunk_container *c, struct cntr_trckr *t, int p, int d, int tc) {
	for (int col = 0; col < d; col++) {
		//aggiunto le coordinate del punto in input al centroide corrispondente;
		set_element_in(t->nwcntr, tc, col, get_element_from(t->nwcntr, tc, col) + get_element_from(c->points, p, col));
	}
	//aggiorno il contatore degli elementi contenuti in ogni centroide;
	t->total[tc] = t->total[tc] + 1;
}

//assegno l'elemento alla mappa; la chiave è il punto; il valore è il centroide più vicino;
void assign_to_map(struct chunk_container *chunk, int point, struct minimum_distance *md) {
	set_element_in(chunk->map, point, M_KEY, point + chunk->offset);
	set_element_in(chunk->map, point, M_VALUE, md->closest_point);
}

//inizializzo il tracker;
struct cntr_trckr *init_tracker(int ncentr, int ndim) {
	struct cntr_trckr *tracker = malloc(sizeof(struct cntr_trckr));
	tracker->nwcntr = alloc_basic_matr(ncentr, ndim);
	tracker->total = alloc_int(ncentr);

	for(int i = 0; i < ncentr; i++) {
		for(int j = 0; j < ndim; j++) {
			set_element_in(tracker->nwcntr, i, j , 0.0);
		}
		tracker->total[i] = 0.0;
	}

	return tracker;
}

struct cntrd_res {
	double j;
	struct cntr_trckr *tracker;
};

struct cntrd_res *assign_centroids(struct chunk_container *chunk) {
	int npoints = get_row(chunk->points->size);
	int ndimensions = get_col(chunk->points->size);
	int ncentroids = get_row(chunk->centroids->size);

	struct cntrd_res *ncntr = malloc(sizeof(struct cntrd_res));
	//questa struttura viene utilizzata per tenere traccia dei punti del centroidi calcolati durante l'iterazione
	//e della somma delle distanze euclidee;
	ncntr->tracker = init_tracker(ncentroids, ndimensions);
	ncntr->j = 0.0;

	//itero sui punti del data frame locale;
	for (int point = 0; point < npoints; point++) {
		//itero sui centroidi calcolati fin'ora
		struct minimum_distance *md = init_md();
		//questa struttura viene utilizzata per tenere traccia del centroide più vicino.
		for (int centroid = 0; centroid < ncentroids; centroid++) {
			//calcolo la distanza tra punto e centroide
			double distance = euclidean_distance(chunk, point, centroid, ndimensions);
			//aggiorno j
			ncntr->j += distance;
			//aggiorno la struttura [SE il centroide è il più vicino trovato fin'ora];
			update_md(md, distance, centroid);
		}
		//assegno il punto al centroide più vicino;
		keep_track(chunk, ncntr->tracker, point, ndimensions, md->closest_point);
		//mantengo l'associazione nella mappa;
		assign_to_map(chunk, point, md);
		if(md) {
			free(md);
		}
	}
	return ncntr;
}

//questa funzione viene utilizzata per calcolare i valori dei centroidi;
struct matrix_container *get_avg(struct cntr_trckr *t) {
	int nrow = get_row(t->nwcntr->size);
	int ncol = get_col(t->nwcntr->size);

	for (int row = 0; row < nrow; row++) {
		for(int col = 0; col < ncol; col++) {
			set_element_in(t->nwcntr, row, col, get_element_from(t->nwcntr, row, col)/((double)t->total[row]));
			//ogni punto del centroide viene aggiornato; dividendolo per il numero di elementi assegnati a quel
			//centroide; in pratica, viene calcolata la media di ogni punto di ogni centroide.
		}
	}

	return t->nwcntr;
}

void kmeans(struct chunk_container *chunk, struct mk_conf *params) {
	//il parametro j viene utilizzato per misurare la convergenza;
	double old_j = 0.0;
	//il parametro max_iterations viene utilizzato per evitare che il metodo cicli all'infinito
	//(anche se è un'ipotesi molto remota)
	for(int i = 0; i < params->max_iterations; i++) {
		//assegno i nuovi centroidi;
		struct cntrd_res *new = assign_centroids(chunk);
		//elimino i precedenti
		free(chunk->centroids);
		//calcolo i nuovi centroidi e li assegno alla struttura principale
		chunk->centroids = get_avg(new->tracker);

		//questa differenza viene utilizzata per verificare che ci sia stata convergenza;
		double diff = old_j - new->j;
		if (diff >= 0.0 && diff < params->threshold) {
			//se c'è stata, il metodo finisce qui;
			return;
		}
		//se no, il nuovo parametro j [ovvero la somma delle distanze euclidee tra i punti e i centroidi]
		//viene assegnato al vecchio j;
		old_j = new->j;
		free(new);
	}
}

void kmeans_init_rand(struct chunk_container *chunk, struct mk_conf *params) {
	//assegno i centroidi iniziali [tramite algoritmo di knuth]
	random_centroids(chunk);
	//applico il kmeans
	kmeans(chunk, params);
}

struct chunk_container *make_chunk_kmeans(double *data, int rows, int cols, int start_line, struct mk_conf *params) {
	struct chunk_container *ch;
	//inizializzo le dimensioni del data frame locale
	int *sizes = to_container(rows, cols);
	//inizializzo la struttura che contiene i dati riguardanti il proprio data frame
	//ovvero la matrice dei dati, la matrice che contiene la mappa e la matrice che contiene i centroidi
	ch = build_chunk_container(data, start_line, sizes, params);
	//parte il metodo del kmeans
	kmeans_init_rand(ch, params);
	return ch;
}

struct matrix_container *join_data(struct chunk_container *chunk){
	int rows = get_row(chunk->points->size);
	int cols = get_col(chunk->points->size);
	struct matrix_container *res = alloc_basic_matr(rows, cols + 1);	
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			set_element_in(res, i, j, get_element_from(chunk->points, i, j));		
		}
		set_element_in(res, i, cols, get_element_from(chunk->map, i, M_VALUE));
	}
	return res;
}

void write_to_csv(double *data, int row, int col, int last_d, char *fname){
	FILE *fp = fopen(fname, "w+");
	struct matrix_container *mc = init_matrix(data, to_container(row,col));
	for (int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			double ele = get_element_from(mc, i , j);
			if ((last_d == 1)&&(j == col-1)){
				fprintf(fp, "%d", (int) ele);
			}else{
				fprintf(fp, "%f", ele);
			}
			if(j < col - 1){
				fprintf(fp, ",");
			}	
		}
		if (i < row - 1){
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void write_centroids_to_csv(struct chunk_container *ch, char *fname){
	write_to_csv(ch->centroids->data, get_row(ch->centroids->size), get_col(ch->centroids->size), 0, fname);
}

void benchmark_it(int procs, int nrows, int ncols, double time){
	printf("%d,%d,%f\n",procs,nrows*ncols,time);
}

int main(int argc, char **argv) {
	char centr_output[] = "centroids.csv";
	char points_final[] = "final_table.csv";

	//estraggo i parametri dati da linea di comando
	struct mk_conf parameters = extract_parameters(argc, argv);
	//estraggo i parametri i parametri mpi [rank, numero di processi]
	struct mpi_state_container state = init_mpi_state(&argc, &argv);
	struct matrix_size m_size;
	struct file_reader r_eadr;
	int *rowcol;
	//il separatore del file csv è la virgola, in futuro verrà dato da linea di comando
	char sep = ',';
	double *data_cnt;
	double *all_centr_cntnr;
	double *final_csv;
	double time_start, time_end, time_max;

	//se il processo è root [ovvero 0]
	if (state.rank == ROOT_PROCESS) {
		if (parameters.filepath == NULL) {
			return abort_app("specify a file [call program with flag --usage]\n", 1);
		}

		//leggo le informazioni riguardanti il file specificato in input
		m_size = init_lengths(parameters.filepath, sep);
		//gestisco vari errori
		if (m_size.err == FILE_NOT_FOUND) {
			char err_msg[80];
			sprintf(err_msg,"file (%s) not found\n", parameters.filepath);
			return abort_app(err_msg, 1);
		}

		if (m_size.rowlen % state.nproc != 0) {
			return abort_app("matrix rows and process number must be divisible\n", 1);
		}

		if (m_size.rowlen/state.nproc < parameters.centroids) {
			char err_msg[80];
			sprintf(err_msg,"centroids must be less than %d, got %d\n", m_size.rowlen/state.nproc, parameters.centroids);
			return abort_app(err_msg, 1);
		}
	}

	// mando le dimensioni del dataframe specificato in input a tutti i processi
	rowcol = broadcast_matrix_size(m_size, state.rank);
	//ogni processo ricava l'offset [ovvero da quale linea comincia a leggere dal file], le linee da leggere
	//e lo spazio da allocare alla propria porzione di data frame
	r_eadr = init_reader(rowcol, &state);
	printf("%d %d %d\n", r_eadr.array_size, r_eadr.rows_to_read, r_eadr.start_line);
	//ogni processo salva la propria prozione di dati [il separatore viene utilizzato per distinguere le colonne]
	data_cnt = init_local_array(&r_eadr, parameters.filepath, sep);
	
	time_start = MPI_Wtime();
	//START THE PARTY
	//qui viene inizializzata la struttura che contiene i dati e poi viene lanciato il metodo del kmeans;
	//assegnando inizialmente centroidi random.
	struct chunk_container *ch = make_chunk_kmeans(data_cnt, r_eadr.rows_to_read, get_col(rowcol), r_eadr.start_line,&parameters);
	
	if((state.nproc == 1) && (benchmark == 1)){
		//se il processo è unico;
		time_end = MPI_Wtime() - time_start;
		//stampa tempi;
		benchmark_it(state.nproc, get_row(ch->points->size), get_col(ch->points->size), time_end);
		//TODO chiama free;
		MPI_Finalize();
		return 0;
	}
	//aspetto che tutti abbiano finito
	MPI_Barrier(MPI_COMM_WORLD);
	
	//ricavo le dimensioni della matrice locale che contiene le coordinate dei centroidi
	int loc_dim = get_row(ch->centroids->size) * get_col(ch->centroids->size);
	//per ricavare la dimensione della matrice globale dei centroidi moltiplico loc_dim per il numero di processi;
	int all_dim = state.nproc * loc_dim;

	//il processo root alloca la dimensione della matrice locale
	if (state.rank == ROOT_PROCESS) {
		all_centr_cntnr = alloc_double(all_dim);
	}

	//tutti i processi mandano al processo root i centroidi ricavati
	MPI_Gather(ch->centroids->data, loc_dim, MPI_DOUBLE, all_centr_cntnr, loc_dim, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);

	//double *global_map;
	struct chunk_container *cc = malloc(sizeof(struct chunk_container));
	cc->map = malloc(sizeof(struct matrix_container));

	cc->centroids = malloc(sizeof(struct matrix_container));
	double *final_centroids;

	if (state.rank == ROOT_PROCESS) {
		//il processo root calcola il kmeans sui centroidi ricevuti
		int rows = state.nproc * get_row(ch->centroids->size);
		int cols = get_col(ch->centroids->size);
		cc = make_chunk_kmeans(all_centr_cntnr, rows, cols, 0, &parameters);
		//la struttura rappresentante la mappa dei centroidi viene assegnata a global_map;
		//la mappa è della forma <ID_PUNTO,ID_CENTROIDE>, per ogni punto si tiene traccia del centroide più vicino
		//quest'operazione viene fatta per aggiornare le mappe locali.
		final_centroids = cc->centroids->data;
		free(cc);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//ricavo le dimensioni della mappa locale
	int num = get_row(ch->centroids->size)*get_col(ch->centroids->size);
	if (state.rank != ROOT_PROCESS) {
		final_centroids = alloc_double(num);
	}

	MPI_Bcast(final_centroids, num, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);
	ch->centroids->data = final_centroids;
	assign_centroids(ch);
	
	if(state.nproc > 1 && benchmark == 1){
		time_end = MPI_Wtime() - time_start;
		MPI_Reduce(&time_end, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		if (state.rank == 0){
			benchmark_it(state.nproc, get_row(rowcol), get_col(rowcol), time_max);
		}
		//TODO chiama free;
		MPI_Finalize();
		return 0;
	}

	struct matrix_container *result = join_data(ch);

	int lc_size = get_row(result->size)*get_col(result->size);
	int gc_size = get_row(rowcol)*get_col(result->size);	
	if (state.rank == 0){
		final_csv = alloc_double(gc_size);
	}

	MPI_Gather(result->data, lc_size, MPI_DOUBLE, final_csv, lc_size, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);
	
	if(state.rank == 0){
		write_to_csv(final_csv,get_row(rowcol), get_col(result->size), 1, points_final);
		write_centroids_to_csv(ch, centr_output);
	}
	
	if (rowcol) {
		free(rowcol);
	}
	if (data_cnt) {
		free(data_cnt);
	}
	MPI_Finalize();
}
