#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "writer.c"


void kat_bytes(unsigned char *bytes, int length) {
    for (int i = 0; i < length; i++) {
        bytes[i] = i % 256;
    }
}

void kat_aead(const char *variant) {
    int MAX_MESSAGE_LENGTH = 32;
    int MAX_ASSOCIATED_DATA_LENGTH = 32;
    int klen = 16;
    int nlen = 16;
    int tlen = 16;
    char filename[50];
    sprintf(filename, "LWC_AEAD_KAT_%d_%d", klen * 8, nlen * 8);
    assert(strcmp(variant, "Ascon-AEAD128") == 0);

    unsigned char key[klen];
    unsigned char nonce[nlen];
    unsigned char msg[MAX_MESSAGE_LENGTH];
    unsigned char ad[MAX_ASSOCIATED_DATA_LENGTH];
    kat_bytes(key, klen);
    kat_bytes(nonce, nlen);
    kat_bytes(msg, MAX_MESSAGE_LENGTH);
    kat_bytes(ad, MAX_ASSOCIATED_DATA_LENGTH);

    MultipleWriter writer;
    multiple_writer_init(&writer, filename);
    int count = 1;
    for (int mlen = 0; mlen <= MAX_MESSAGE_LENGTH; mlen++) {
        for (int adlen = 0; adlen <= MAX_ASSOCIATED_DATA_LENGTH; adlen++) {
            multiple_writer_open(&writer);
            char count_str[10];
            sprintf(count_str, "%d", count);
            multiple_writer_append(&writer, "Count", count_str);
            count++;
            multiple_writer_append(&writer, "Key", (char *)key);
            multiple_writer_append(&writer, "Nonce", (char *)nonce);
            multiple_writer_append(&writer, "PT", (char *)msg);
            multiple_writer_append(&writer, "AD", (char *)ad);
            unsigned char ct[mlen + tlen];
            ascon_encrypt(key, nonce, ad, adlen, msg, mlen, ct, tlen);
            multiple_writer_append(&writer, "CT", (char *)ct);
            unsigned char msg2[mlen];
            ascon_decrypt(key, nonce, ad, adlen, ct, mlen + tlen, msg2, variant);
            assert(memcmp(msg2, msg, mlen) == 0);
            multiple_writer_close(&writer);
        }
    }
}

void kat_hash(const char *variant) {
    int MAX_MESSAGE_LENGTH = 1024;
    int hlen = 32;
    char filename[50];
    sprintf(filename, "LWC_HASH_KAT_%d", hlen * 8);
    assert(strcmp(variant, "Ascon-Hash256") == 0 || strcmp(variant, "Ascon-XOF128") == 0 || strcmp(variant, "Ascon-CXOF128") == 0);

    unsigned char msg[MAX_MESSAGE_LENGTH];
    kat_bytes(msg, MAX_MESSAGE_LENGTH);

    MultipleWriter writer;
    multiple_writer_init(&writer, filename);
    int count = 1;
    for (int mlen = 0; mlen <= MAX_MESSAGE_LENGTH; mlen++) {
        multiple_writer_open(&writer);
        char count_str[10];
        sprintf(count_str, "%d", count);
        multiple_writer_append(&writer, "Count", count_str);
        count++;
        multiple_writer_append(&writer, "Msg", (char *)msg);
        unsigned char tag[hlen];
        ascon_hash(msg, mlen, tag, variant);
        multiple_writer_append(&writer, "MD", (char *)tag);
        multiple_writer_close(&writer);
    }
}

void kat(const char *variant) {
    if (strcmp(variant, "Ascon-AEAD128") == 0) {
        kat_aead(variant);
    } else if (strcmp(variant, "Ascon-Hash256") == 0 || strcmp(variant, "Ascon-XOF128") == 0 || strcmp(variant, "Ascon-CXOF128") == 0) {
        kat_hash(variant);
    } else {
        fprintf(stderr, "Unknown variant: %s\n", variant);
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    const char *variant = argc > 1 ? argv[1] : "Ascon-AEAD128";
    kat(variant);
    return 0;
}