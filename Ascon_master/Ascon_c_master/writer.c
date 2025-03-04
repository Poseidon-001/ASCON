#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct {
    FILE *fp;
    int is_open;
} TextWriter;

typedef struct {
    FILE *fp;
    int level;
    int has_item;
    char *tab;
} JSONWriter;

typedef struct {
    TextWriter *text_writer;
    JSONWriter *json_writer;
} MultipleWriter;

// TextWriter functions
TextWriter* TextWriter_new(const char *filename) {
    TextWriter *writer = (TextWriter*)malloc(sizeof(TextWriter));
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s.txt", filename);
    writer->fp = fopen(filepath, "w");
    writer->is_open = 0;
    return writer;
}

void TextWriter_open(TextWriter *writer) {
    assert(!writer->is_open && "cannot open twice");
    writer->is_open = 1;
}

void TextWriter_append(TextWriter *writer, const char *label, const char *value, int length) {
    assert(writer->is_open && "cannot append if not open yet");
    if (length != -1) {
        assert(strlen(value) >= length);
        char truncated_value[256];
        strncpy(truncated_value, value, length);
        truncated_value[length] = '\0';
        fprintf(writer->fp, "%s = %s\n", label, truncated_value);
    } else {
        fprintf(writer->fp, "%s = %s\n", label, value);
    }
}

void TextWriter_close(TextWriter *writer) {
    assert(writer->is_open && "cannot close if not open first");
    fprintf(writer->fp, "\n");
    writer->is_open = 0;
    fclose(writer->fp);
    free(writer);
}

// JSONWriter functions
JSONWriter* JSONWriter_new(const char *filename) {
    JSONWriter *writer = (JSONWriter*)malloc(sizeof(JSONWriter));
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s.json", filename);
    writer->fp = fopen(filepath, "w");
    writer->level = 1;
    writer->has_item = 0;
    writer->tab = "  ";
    fprintf(writer->fp, "[");
    return writer;
}

void JSONWriter_open(JSONWriter *writer) {
    assert((writer->level > 0 || !writer->has_item) && "cannot open twice");
    fprintf(writer->fp, "%s%s{", writer->has_item ? "," : "", writer->tab);
    writer->level++;
    writer->has_item = 0;
}

void JSONWriter_append(JSONWriter *writer, const char *label, const char *value, int length) {
    if (length != -1) {
        assert(strlen(value) >= length);
        char truncated_value[256];
        strncpy(truncated_value, value, length);
        truncated_value[length] = '\0';
        fprintf(writer->fp, "%s\n%s\"%s\": \"%s\"", writer->has_item ? "," : "", writer->tab, label, truncated_value);
    } else {
        fprintf(writer->fp, "%s\n%s\"%s\": \"%s\"", writer->has_item ? "," : "", writer->tab, label, value);
    }
    writer->has_item = 1;
}

void JSONWriter_close(JSONWriter *writer) {
    assert((writer->level > 0 || !writer->has_item) && "cannot close if not open first");
    writer->level--;
    fprintf(writer->fp, "%s}\n", writer->tab);
    writer->has_item = 1;
    fprintf(writer->fp, "]\n");
    fclose(writer->fp);
    free(writer);
}

// MultipleWriter functions
MultipleWriter* MultipleWriter_new(const char *filename) {
    MultipleWriter *writer = (MultipleWriter*)malloc(sizeof(MultipleWriter));
    writer->text_writer = TextWriter_new(filename);
    writer->json_writer = JSONWriter_new(filename);
    return writer;
}

void MultipleWriter_open(MultipleWriter *writer) {
    TextWriter_open(writer->text_writer);
    JSONWriter_open(writer->json_writer);
}

void MultipleWriter_append(MultipleWriter *writer, const char *label, const char *value, int length) {
    TextWriter_append(writer->text_writer, label, value, length);
    JSONWriter_append(writer->json_writer, label, value, length);
}

void MultipleWriter_close(MultipleWriter *writer) {
    TextWriter_close(writer->text_writer);
    JSONWriter_close(writer->json_writer);
    free(writer);
}

int main() {
    MultipleWriter *writer = MultipleWriter_new("demo");
    MultipleWriter_open(writer);
    MultipleWriter_append(writer, "Hello", "101", -1);
    MultipleWriter_close(writer);
    return 0;
}