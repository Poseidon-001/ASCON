#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#define debug 0
#define debugpermutation 0

void ascon_permutation(uint64_t *S, int rounds);

void zero_bytes(uint8_t *bytes, size_t n) {
    memset(bytes, 0, n);
}

void ff_bytes(uint8_t *bytes, size_t n) {
    memset(bytes, 0xFF, n);
}

void to_bytes(uint8_t *dest, const uint8_t *src, size_t len) {
    memcpy(dest, src, len);
}

uint64_t bytes_to_int(const uint8_t *bytes, size_t len) {
    uint64_t result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= ((uint64_t)bytes[i]) << (i * 8);
    }
    return result;
}

void bytes_to_state(uint64_t *state, const uint8_t *bytes) {
    for (size_t i = 0; i < 5; i++) {
        state[i] = bytes_to_int(bytes + 8 * i, 8);
    }
}

void int_to_bytes(uint8_t *bytes, uint64_t integer, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) {
        bytes[i] = (integer >> (i * 8)) & 0xFF;
    }
}

uint64_t rotr(uint64_t val, int r) {
    return (val >> r) | (val << (64 - r));
}

void printstate(const uint64_t *state, const char *description) {
    printf(" %s\n", description);
    for (size_t i = 0; i < 5; i++) {
        printf("%016lx ", state[i]);
    }
    printf("\n");
}

void printwords(const uint64_t *state, const char *description) {
    printf(" %s\n", description);
    for (size_t i = 0; i < 5; i++) {
        printf("  x%zu=%016lx\n", i, state[i]);
    }
}  

void ascon_permutation(uint64_t *S, int rounds);

void ascon_permutation(uint64_t *S, int rounds) {
    assert(rounds <= 12);
    if (debugpermutation) printwords(S, "permutation input:");
    for (int r = 12 - rounds; r < 12; r++) {
        // --- add round constants ---
        S[2] ^= (0xf0 - r * 0x10 + r * 0x1);
        if (debugpermutation) printwords(S, "round constant addition:");
        // --- substitution layer ---
        S[0] ^= S[4];
        S[4] ^= S[3];
        S[2] ^= S[1];
        uint64_t T[5];
        for (int i = 0; i < 5; i++) {
            T[i] = (S[i] ^ 0xFFFFFFFFFFFFFFFF) & S[(i + 1) % 5];
        }
        for (int i = 0; i < 5; i++) {
            S[i] ^= T[(i + 1) % 5];
        }
        S[1] ^= S[0];
        S[0] ^= S[4];
        S[3] ^= S[2];
        S[2] ^= 0xFFFFFFFFFFFFFFFF;
        if (debugpermutation) printwords(S, "substitution layer:");
        // --- linear diffusion layer ---
        S[0] ^= rotr(S[0], 19) ^ rotr(S[0], 28);
        S[1] ^= rotr(S[1], 61) ^ rotr(S[1], 39);
        S[2] ^= rotr(S[2],  1) ^ rotr(S[2],  6);
        S[3] ^= rotr(S[3], 10) ^ rotr(S[3], 17);
        S[4] ^= rotr(S[4],  7) ^ rotr(S[4], 41);
        if (debugpermutation) printwords(S, "linear diffusion layer:");
    }
}

void ascon_hash(uint8_t *hash, const uint8_t *message, size_t message_len, const char *variant, size_t hashlength, const uint8_t *customization, size_t customization_len) {
    uint8_t versions[] = {2, 3, 4};
    int variant_index = -1;
    if (strcmp(variant, "Ascon-Hash256") == 0) variant_index = 0;
    else if (strcmp(variant, "Ascon-XOF128") == 0) variant_index = 1;
    else if (strcmp(variant, "Ascon-CXOF128") == 0) variant_index = 2;
    assert(variant_index != -1);
    if (variant_index == 0) assert(hashlength == 32);
    if (variant_index == 2) assert(customization_len <= 256);
    else assert(customization_len == 0);

    int a = 12, b = 12;
    int rate = 8;
    int taglen = (variant_index == 0) ? 256 : 0;
    int customize = (variant_index == 2) ? 1 : 0;

    uint8_t iv[40];
    iv[0] = versions[variant_index];
    iv[1] = 0;
    iv[2] = (b << 4) + a;
    int_to_bytes(iv + 3, taglen, 2);
    iv[5] = rate;
    iv[6] = 0;
    iv[7] = 0;
    zero_bytes(iv + 8, 32);

    uint64_t S[5];
    bytes_to_state(S, iv);
    if (debug) printstate(S, "initial value:");

    ascon_permutation(S, 12);
    if (debug) printstate(S, "initialization:");

    if (customize) {
        size_t z_padding_len = rate - (customization_len % rate) - 1;
        uint8_t z_padding[z_padding_len + 1];
        z_padding[0] = 0x01;
        zero_bytes(z_padding + 1, z_padding_len);

        uint8_t z_length[8];
        int_to_bytes(z_length, customization_len * 8, 8);

        uint8_t z_padded[8 + customization_len + z_padding_len + 1];
        memcpy(z_padded, z_length, 8);
        memcpy(z_padded + 8, customization, customization_len);
        memcpy(z_padded + 8 + customization_len, z_padding, z_padding_len + 1);

        for (size_t block = 0; block < 8 + customization_len + z_padding_len + 1; block += rate) {
            S[0] ^= bytes_to_int(z_padded + block, rate);
            ascon_permutation(S, 12);
        }
        if (debug) printstate(S, "customization:");
    }

    size_t m_padding_len = rate - (message_len % rate) - 1;
    uint8_t m_padding[m_padding_len + 1];
    m_padding[0] = 0x01;
    zero_bytes(m_padding + 1, m_padding_len);

    uint8_t m_padded[message_len + m_padding_len + 1];
    memcpy(m_padded, message, message_len);
    memcpy(m_padded + message_len, m_padding, m_padding_len + 1);

    for (size_t block = 0; block < message_len + m_padding_len + 1; block += rate) {
        S[0] ^= bytes_to_int(m_padded + block, rate);
        ascon_permutation(S, 12);
    }
    if (debug) printstate(S, "process message:");

    size_t H_len = 0;
    while (H_len < hashlength) {
        int_to_bytes(hash + H_len, S[0], rate);
        H_len += rate;
        ascon_permutation(S, 12);
    }
    if (debug) printstate(S, "finalization:");
}

void ascon_mac(uint8_t *tag, const uint8_t *key, const uint8_t *message, size_t message_len, const char *variant, size_t taglength) {
    assert(strcmp(variant, "Ascon-Mac") == 0 || strcmp(variant, "Ascon-Prf") == 0 || strcmp(variant, "Ascon-PrfShort") == 0);
    if (strcmp(variant, "Ascon-Mac") == 0) assert(strlen((char *)key) == 16 && taglength <= 16);
    if (strcmp(variant, "Ascon-Prf") == 0) assert(strlen((char *)key) == 16);
    if (strcmp(variant, "Ascon-PrfShort") == 0) assert(strlen((char *)key) == 16 && taglength <= 16 && message_len <= 16);

    int a = 12, b = 12;
    int msgblocksize = 32;
    int rate = 16;

    uint64_t S[5];

    if (strcmp(variant, "Ascon-PrfShort") == 0) {
        uint8_t IV[20];
        IV[0] = strlen((char *)key) * 8;
        IV[1] = message_len * 8;
        IV[2] = a + 64;
        IV[3] = taglength * 8;
        zero_bytes(IV + 4, 16);

        uint8_t state_bytes[48];
        memcpy(state_bytes, IV, 20);
        memcpy(state_bytes + 20, key, 16);
        memcpy(state_bytes + 36, message, message_len);
        zero_bytes(state_bytes + 36 + message_len, 16 - message_len);

        bytes_to_state(S, state_bytes);
        if (debug) printstate(S, "initial value:");

        ascon_permutation(S, a);
        if (debug) printstate(S, "process message:");

        uint8_t T[16];
        int_to_bytes(T, S[3] ^ bytes_to_int(key, 8), 8);
        int_to_bytes(T + 8, S[4] ^ bytes_to_int(key + 8, 8), 8);
        memcpy(tag, T, taglength);
    } else {
        uint8_t tagspec[4];
        if (strcmp(variant, "Ascon-Mac") == 0) int_to_bytes(tagspec, 16 * 8, 4);
        if (strcmp(variant, "Ascon-Prf") == 0) int_to_bytes(tagspec, 0 * 8, 4);

        uint8_t state_bytes[48];
        state_bytes[0] = strlen((char *)key) * 8;
        state_bytes[1] = rate * 8;
        state_bytes[2] = a + 128;
        state_bytes[3] = a - b;
        memcpy(state_bytes + 4, tagspec, 4);
        memcpy(state_bytes + 8, key, 16);
        zero_bytes(state_bytes + 24, 24);

        bytes_to_state(S, state_bytes);
        if (debug) printstate(S, "initial value:");

        ascon_permutation(S, a);
        if (debug) printstate(S, "initialization:");

        size_t m_padding_len = msgblocksize - (message_len % msgblocksize) - 1;
        uint8_t m_padding[m_padding_len + 1];
        m_padding[0] = 0x01;
        zero_bytes(m_padding + 1, m_padding_len);

        uint8_t m_padded[message_len + m_padding_len + 1];
        memcpy(m_padded, message, message_len);
        memcpy(m_padded + message_len, m_padding, m_padding_len + 1);

        for (size_t block = 0; block < message_len + m_padding_len + 1 - msgblocksize; block += msgblocksize) {
            S[0] ^= bytes_to_int(m_padded + block, 8);
            S[1] ^= bytes_to_int(m_padded + block + 8, 8);
            S[2] ^= bytes_to_int(m_padded + block + 16, 8);
            S[3] ^= bytes_to_int(m_padded + block + 24, 8);
            ascon_permutation(S, b);
        }
        size_t block = message_len + m_padding_len + 1 - msgblocksize;
        S[0] ^= bytes_to_int(m_padded + block, 8);
        S[1] ^= bytes_to_int(m_padded + block + 8, 8);
        S[2] ^= bytes_to_int(m_padded + block + 16, 8);
        S[3] ^= bytes_to_int(m_padded + block + 24, 8);
        S[4] ^= 1;
        if (debug) printstate(S, "process message:");

        uint8_t T[32];
        size_t T_len = 0;
        ascon_permutation(S, a);
        while (T_len < taglength) {
            int_to_bytes(T + T_len, S[0], 8);
            T_len += 8;
            int_to_bytes(T + T_len, S[1], 8);
            T_len += 8;
            ascon_permutation(S, b);
        }
        if (debug) printstate(S, "finalization:");
        memcpy(tag, T, taglength);
    }
}

void ascon_encrypt(uint8_t *ciphertext, const uint8_t *key, const uint8_t *nonce, const uint8_t *associateddata, size_t adlen, const uint8_t *plaintext, size_t ptlen, const char *variant) {
    uint8_t versions[] = {1};
    int variant_index = -1;
    if (strcmp(variant, "Ascon-AEAD128") == 0) variant_index = 0;
    assert(variant_index != -1);
    assert(key != NULL && nonce != NULL);

    int a = 12, b = 8;
    int rate = 8;
    int taglen = 16;

    uint8_t iv[40];
    iv[0] = versions[variant_index];
    iv[1] = 0;
    iv[2] = (b << 4) + a;
    int_to_bytes(iv + 3, taglen, 2);
    iv[5] = rate;
    iv[6] = 0;
    iv[7] = 0;
    memcpy(iv + 8, key, 16);
    memcpy(iv + 24, nonce, 16);
    zero_bytes(iv + 40, 16);

    uint64_t S[5];
    bytes_to_state(S, iv);
    if (debug) printstate(S, "initial value:");

    ascon_permutation(S, a);
    if (debug) printstate(S, "initialization:");

    if (adlen > 0) {
        size_t a_padding_len = rate - (adlen % rate) - 1;
        uint8_t a_padding[a_padding_len + 1];
        a_padding[0] = 0x80;
        zero_bytes(a_padding + 1, a_padding_len);

        uint8_t a_padded[adlen + a_padding_len + 1];
        memcpy(a_padded, associateddata, adlen);
        memcpy(a_padded + adlen, a_padding, a_padding_len + 1);

        for (size_t block = 0; block < adlen + a_padding_len + 1; block += rate) {
            S[0] ^= bytes_to_int(a_padded + block, rate);
            ascon_permutation(S, b);
        }
        if (debug) printstate(S, "process associated data:");
    }

    S[4] ^= 1;
    if (debug) printstate(S, "domain separation:");

    size_t p_padding_len = rate - (ptlen % rate) - 1;
    uint8_t p_padding[p_padding_len + 1];
    p_padding[0] = 0x80;
    zero_bytes(p_padding + 1, p_padding_len);

    uint8_t p_padded[ptlen + p_padding_len + 1];
    memcpy(p_padded, plaintext, ptlen);
    memcpy(p_padded + ptlen, p_padding, p_padding_len + 1);

    size_t c_len = 0;
    for (size_t block = 0; block < ptlen + p_padding_len + 1; block += rate) {
        S[0] ^= bytes_to_int(p_padded + block, rate);
        int_to_bytes(ciphertext + c_len, S[0], rate);
        c_len += rate;
        ascon_permutation(S, b);
    }
    c_len = ptlen;
    if (debug) printstate(S, "process plaintext:");

    S[1] ^= bytes_to_int(key, 8);
    S[2] ^= bytes_to_int(key + 8, 8);
    ascon_permutation(S, a);
    S[3] ^= bytes_to_int(key, 8);
    S[4] ^= bytes_to_int(key + 8, 8);
    if (debug) printstate(S, "finalization:");

    uint8_t tag[16];
    int_to_bytes(tag, S[3], 8);
    int_to_bytes(tag + 8, S[4], 8);
    memcpy(ciphertext + ptlen, tag, taglen);

    if (debug) {
        printf("Generated tag: ");
        for (int i = 0; i < taglen; i++) {
            printf("%02x", tag[i]);
        }
        printf("\n");
    }
}

void ascon_decrypt(uint8_t *plaintext, const uint8_t *key, const uint8_t *nonce, const uint8_t *associateddata, size_t adlen, const uint8_t *ciphertext, size_t ctlen, const char *variant) {
    uint8_t versions[] = {1};
    int variant_index = -1;
    if (strcmp(variant, "Ascon-AEAD128") == 0) variant_index = 0;
    assert(variant_index != -1);
    assert(key != NULL && nonce != NULL && ctlen >= 16);

    int a = 12, b = 8;
    int rate = 8;
    int taglen = 16;

    uint8_t iv[40];
    iv[0] = versions[variant_index];
    iv[1] = 0;
    iv[2] = (b << 4) + a;
    int_to_bytes(iv + 3, taglen, 2);
    iv[5] = rate;
    iv[6] = 0;
    iv[7] = 0;
    memcpy(iv + 8, key, 16);
    memcpy(iv + 24, nonce, 16);
    zero_bytes(iv + 40, 16);

    uint64_t S[5];
    bytes_to_state(S, iv);
    if (debug) printstate(S, "initial value:");

    ascon_permutation(S, a);
    if (debug) printstate(S, "initialization:");

    if (adlen > 0) {
        size_t a_padding_len = rate - (adlen % rate) - 1;
        uint8_t a_padding[a_padding_len + 1];
        a_padding[0] = 0x80;
        zero_bytes(a_padding + 1, a_padding_len);

        uint8_t a_padded[adlen + a_padding_len + 1];
        memcpy(a_padded, associateddata, adlen);
        memcpy(a_padded + adlen, a_padding, a_padding_len + 1);

        for (size_t block = 0; block < adlen + a_padding_len + 1; block += rate) {
            S[0] ^= bytes_to_int(a_padded + block, rate);
            ascon_permutation(S, b);
        }
        if (debug) printstate(S, "process associated data:");
    }

    S[4] ^= 1;
    if (debug) printstate(S, "domain separation:");

    size_t c_padding_len = rate - ((ctlen - taglen) % rate) - 1;
    uint8_t c_padding[c_padding_len + 1];
    c_padding[0] = 0x80;
    zero_bytes(c_padding + 1, c_padding_len);

    uint8_t c_padded[ctlen - taglen + c_padding_len + 1];
    memcpy(c_padded, ciphertext, ctlen - taglen);
    memcpy(c_padded + ctlen - taglen, c_padding, c_padding_len + 1);

    size_t p_len = 0;
    for (size_t block = 0; block < ctlen - taglen + c_padding_len + 1; block += rate) {
        uint64_t c_block = bytes_to_int(c_padded + block, rate);
        int_to_bytes(plaintext + p_len, S[0] ^ c_block, rate);
        p_len += rate;
        S[0] = c_block;
        ascon_permutation(S, b);
    }
    p_len = ctlen - taglen;
    if (debug) printstate(S, "process ciphertext:");

    S[1] ^= bytes_to_int(key, 8);
    S[2] ^= bytes_to_int(key + 8, 8);
    ascon_permutation(S, a);
    S[3] ^= bytes_to_int(key, 8);
    S[4] ^= bytes_to_int(key + 8, 8);
    if (debug) printstate(S, "finalization:");

    uint8_t tag[16];
    int_to_bytes(tag, S[3], 8);
    int_to_bytes(tag + 8, S[4], 8);

    if (debug) {
        printf("Expected tag: ");
        for (int i = 0; i < taglen; i++) {
            printf("%02x", ciphertext[ctlen - taglen + i]);
        }
        printf("\n");
        printf("Computed tag: ");
        for (int i = 0; i < taglen; i++) {
            printf("%02x", tag[i]);
        }
        printf("\n");
    }
}

void ascon_initialize(uint64_t *S, const uint8_t *key, const uint8_t *nonce, int a, int b, int rate, int taglen, uint8_t version) {
    uint8_t iv[40];
    iv[0] = version;
    iv[1] = 0;
    iv[2] = (b << 4) + a;
    int_to_bytes(iv + 3, taglen, 2);
    iv[5] = rate;
    iv[6] = 0;
    iv[7] = 0;
    memcpy(iv + 8, key, 16);
    memcpy(iv + 24, nonce, 16);
    zero_bytes(iv + 40, 16);

    bytes_to_state(S, iv);
    if (debug) printstate(S, "initial value:");

    ascon_permutation(S, a);
    if (debug) printstate(S, "initialization:");
}

void ascon_process_associated_data(uint64_t *S, const uint8_t *associateddata, size_t adlen, int b, int rate) {
    if (adlen > 0) {
        size_t a_padding_len = rate - (adlen % rate) - 1;
        uint8_t a_padding[a_padding_len + 1];
        a_padding[0] = 0x80;
        zero_bytes(a_padding + 1, a_padding_len);

        uint8_t a_padded[adlen + a_padding_len + 1];
        memcpy(a_padded, associateddata, adlen);
        memcpy(a_padded + adlen, a_padding, a_padding_len + 1);

        for (size_t block = 0; block < adlen + a_padding_len + 1; block += rate) {
            S[0] ^= bytes_to_int(a_padded + block, rate);
            ascon_permutation(S, b);
        }
        if (debug) printstate(S, "process associated data:");
    }

    S[4] ^= 1;
    if (debug) printstate(S, "domain separation:");
}

void ascon_process_plaintext(uint64_t *S, uint8_t *ciphertext, const uint8_t *plaintext, size_t ptlen, int b, int rate) {
    size_t p_padding_len = rate - (ptlen % rate) - 1;
    uint8_t p_padding[p_padding_len + 1];
    p_padding[0] = 0x80;
    zero_bytes(p_padding + 1, p_padding_len);

    uint8_t p_padded[ptlen + p_padding_len + 1];
    memcpy(p_padded, plaintext, ptlen);
    memcpy(p_padded + ptlen, p_padding, p_padding_len + 1);

    size_t c_len = 0;
    for (size_t block = 0; block < ptlen + p_padding_len + 1; block += rate) {
        S[0] ^= bytes_to_int(p_padded + block, rate);
        int_to_bytes(ciphertext + c_len, S[0], rate);
        c_len += rate;
        ascon_permutation(S, b);
    }
    if (debug) printstate(S, "process plaintext:");
}

void ascon_process_ciphertext(uint64_t *S, uint8_t *plaintext, const uint8_t *ciphertext, size_t ctlen, int b, int rate) {
    size_t c_padding_len = rate - (ctlen % rate) - 1;
    uint8_t c_padding[c_padding_len + 1];
    c_padding[0] = 0x80;
    zero_bytes(c_padding + 1, c_padding_len);

    uint8_t c_padded[ctlen + c_padding_len + 1];
    memcpy(c_padded, ciphertext, ctlen);
    memcpy(c_padded + ctlen, c_padding, c_padding_len + 1);

    size_t p_len = 0;
    for (size_t block = 0; block < ctlen + c_padding_len + 1; block += rate) {
        uint64_t c_block = bytes_to_int(c_padded + block, rate);
        int_to_bytes(plaintext + p_len, S[0] ^ c_block, rate);
        p_len += rate;
        S[0] = c_block;
        ascon_permutation(S, b);
    }
    if (debug) printstate(S, "process ciphertext:");
}

void ascon_finalize(uint64_t *S, uint8_t *tag, const uint8_t *key, int a, int taglen) {
    S[1] ^= bytes_to_int(key, 8);
    S[2] ^= bytes_to_int(key + 8, 8);
    ascon_permutation(S, a);
    S[3] ^= bytes_to_int(key, 8);
    S[4] ^= bytes_to_int(key + 8, 8);
    if (debug) printstate(S, "finalization:");

    int_to_bytes(tag, S[3], 8);
    int_to_bytes(tag + 8, S[4], 8);
}

void demo_print(const char *text, const uint8_t *val, size_t length) {
    printf("%s: 0x", text);
    for (size_t i = 0; i < length; i++) {
        printf("%02x", val[i]);
    }
    printf(" (%zu bytes)\n", length);
}

void get_random_bytes(uint8_t *bytes, size_t num) {
    for (size_t i = 0; i < num; i++) {
        bytes[i] = rand() % 256;
    }
}

void demo_aead(const char *variant) {
    assert(strcmp(variant, "Ascon-AEAD128") == 0);
    printf("=== demo encryption using %s ===\n", variant);

    uint8_t key[16];
    uint8_t nonce[16];
    uint8_t associateddata[] = "ASCON";
    uint8_t plaintext[] = "ascon";
    uint8_t ciphertext[sizeof(plaintext) + 16];
    uint8_t receivedplaintext[sizeof(plaintext)];

    get_random_bytes(key, 16);
    get_random_bytes(nonce, 16);

    ascon_encrypt(ciphertext, key, nonce, associateddata, sizeof(associateddata) - 1, plaintext, sizeof(plaintext) - 1, variant);
    ascon_decrypt(receivedplaintext, key, nonce, associateddata, sizeof(associateddata) - 1, ciphertext, sizeof(ciphertext), variant);

    demo_print("key", key, 16);
    demo_print("nonce", nonce, 16);
    demo_print("plaintext", plaintext, sizeof(plaintext) - 1);
    demo_print("ass.data", associateddata, sizeof(associateddata) - 1);
    demo_print("ciphertext", ciphertext, sizeof(ciphertext) - 16);
    demo_print("tag", ciphertext + sizeof(ciphertext) - 16, 16);
    demo_print("received", receivedplaintext, sizeof(receivedplaintext) - 1);
}

void demo_hash(const char *variant, size_t hashlength) {
    assert(strcmp(variant, "Ascon-Hash256") == 0 || strcmp(variant, "Ascon-XOF128") == 0 || strcmp(variant, "Ascon-CXOF128") == 0);
    printf("=== demo hash using %s ===\n", variant);

    uint8_t message[] = "ascon";
    uint8_t customization[] = "custom";
    uint8_t tag[hashlength];

    if (strcmp(variant, "Ascon-CXOF128") == 0) {
        ascon_hash(tag, message, sizeof(message) - 1, variant, hashlength, customization, sizeof(customization) - 1);
    } else {
        ascon_hash(tag, message, sizeof(message) - 1, variant, hashlength, NULL, 0);
    }

    demo_print("message", message, sizeof(message) - 1);
    if (strcmp(variant, "Ascon-CXOF128") == 0) {
        demo_print("customization", customization, sizeof(customization) - 1);
    }
    demo_print("tag", tag, hashlength);
}

void demo_mac(const char *variant, size_t taglength) {
    assert(strcmp(variant, "Ascon-Mac") == 0 || strcmp(variant, "Ascon-Prf") == 0 || strcmp(variant, "Ascon-PrfShort") == 0);
    printf("=== demo MAC using %s ===\n", variant);

    uint8_t key[16];
    uint8_t message[] = "ascon";
    uint8_t tag[taglength];

    get_random_bytes(key, 16);

    ascon_mac(tag, key, message, sizeof(message) - 1, variant, taglength);

    demo_print("key", key, 16);
    demo_print("message", message, sizeof(message) - 1);
    demo_print("tag", tag, taglength);
}

int main() {
    srand(time(NULL));
    demo_aead("Ascon-AEAD128");
    demo_hash("Ascon-Hash256", 32);
    demo_hash("Ascon-XOF128", 32);
    demo_hash("Ascon-CXOF128", 32);
    // demo_mac("Ascon-Mac", 16);
    return 0;
}
