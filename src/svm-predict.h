#pragma once
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"


static int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

static struct svm_node *x;
static int max_nr_attr = 64;

static struct svm_model* model;
static int predict_probability=0;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input);
void exit_input_error(int line_num);
void predict(FILE *input, FILE *output);
void exit_with_help();
int svmPredict(int argc, char **argv);
