#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define SEM_INITIAL_VALUE 1
#define SEM_NAME "/my.sem"

#define SHM_SIZE (8192 * 1024UL)
#define SHM_NAME "/my.shm"

#define PERMS (S_IRWXU | S_IRWXG | S_IRWXO)

int main() {
    int rc = 0;
    int fd = shm_open(SHM_NAME, O_RDONLY, PERMS);  /* empty to begin */
    if (fd < 0) {
        perror("shm_open()");
        goto err_shm_open;
    }

    char* memptr = (char*)mmap(NULL, SHM_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (memptr == MAP_FAILED) {
        perror("mmap()");
        goto err_memptr;
    }

    /* create a semaphore for mutual exclusion */
    sem_t* semptr = sem_open(SEM_NAME, O_RDWR);
    if (semptr == SEM_FAILED) {
        perror("sem_open()");
        rc = -1;
        goto err_sem_open;
    }

    printf("sem_wait()'ing\n");
    if (sem_wait(semptr) < 0) {
        perror("sem_wait()");
        rc = -1;
        goto err_semptr;
    }
    
    size_t image_size;    
    memcpy(&image_size, memptr, sizeof(size_t));    
    printf("image_size: %lu\n", image_size);

    char* image_data = (char*)malloc(image_size * sizeof(uint8_t));
    if (image_data == NULL) {
        perror("malloc()");
        rc = -1;
        goto err_malloc;
    }
    memcpy(image_data, memptr + sizeof(size_t), image_size);
    if (sem_post(semptr) != 0) {
        perror("sem_post()");
        rc = -1;
    }
    
    FILE *write_ptr = fopen("/tmp/test.jpg", "wb");
    if (write_ptr == NULL) {
        perror("fopen()");
        rc = -1;
        goto err_fopen;
    }
    if (fwrite(image_data, image_size, 1, write_ptr) != 1) {
        perror("fwrite()");
        rc = -1;
    } else {
        printf("fwrite() succeeded\n");
    }

    /* C-style tedious cleanup */
    if (fclose(write_ptr) != 0) { perror("fclose()"); }
err_fopen:
    free(image_data);
err_malloc:
err_semptr:
    if (sem_close(semptr) != 0) { perror("sem_close()"); }
    // On the reader side, we should not call sem_unlink(); otherwise the named
    // semaphore will be disassociated with the semaphore specified by the
    // name, causing future sem_open() to fail.
err_sem_open:
    if (munmap(memptr, SHM_SIZE) != 0) { perror("munmap()"); }
err_memptr:
    if (close(fd) != 0) { perror("close()"); }
    // On the reader side, we should not call shm_unlink(); otherwise the named
    // shared object will be disassociated with the shared memory object
    // specified by the name, causing future shm_open() to fail.
    // Note that we can force shm_open() to be successful by specifying O_CREAT
    // but by doing so we are creating a new and irrelevant shared memory object.
err_shm_open:
    return rc;
}