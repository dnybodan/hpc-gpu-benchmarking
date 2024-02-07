// Note make sure this is what you want and not the class of other file
// manipulation functions provided by the c standard library such as fopen, and fwrite
// This file explores the more low-level POSIX system open() method because we
// are learning GDS and at a low-level GDS is taking advantage of these system
// calls to be POSIX compatible.
//
// from man 3 open
// The O_DIRECT, O_NOATIME, O_PATH, and O_TMPFILE  flags  are  Linux-specific.   One  must  define
//  _GNU_SOURCE to obtain their definitions.
//
// As noted in feature_test_macros(7), feature test macros such as _POSIX_C_SOURCE, _XOPEN_SOURCE,
// and _GNU_SOURCE must be defined before including any header files.
//
// There is this gem in the programmers manual:
//   In summary, O_DIRECT is a potentially powerful tool that should be used with
//   caution.   It  is recommended  that  applications treat use of O_DIRECT as a
//   performance option which is disabled by default.
//   
//   "The thing that has always disturbed me about O_DIRECT is that the
//   whole  interface  is just  stupid,  and  was probably designed by a deranged
//   monkey on some serious mind-con‐ trolling substances."—Linus
//
// I guess one question I have is what options does nvcc have by default so that
// the nvcc compile of cpp code works fine?

// must be included before any other headers, or add -D_GNU_SOURCE in compiler
// options e.g., gcc -D_GNU_SOURCE ...
//#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
// this would also work, but looks very non-standard... this was me just trying
// to figure out how to get O_DIRECT and found it in
// /usr/include/bits/fcntl-bits.h and needed __USE_GNU defined.
#define __USE_GNU
#include <fcntl.h>

#include <unistd.h>

int main() {

  int fd;
  const char myfile[64] = "helloworld.txt\0";

  fd = open(myfile, O_CREAT | O_WRONLY | O_DIRECT, 0644);
  if(fd < 0) {
    fprintf(stderr, "file open %s errno %d\n", myfile, errno);
    return -1;
  }
  printf("the file descriptor is %d\n", fd);

  // To use O_DIRECT needed to use a posix align to make sure we can write
  // Align the buffer to the block size, typically 512 bytes or a multiple thereof
  // may need to experiment to know beegfs block size...?
  const size_t blockSize = 512;
  char *buf;
  if (posix_memalign((void **)&buf, blockSize, blockSize) != 0) {
      perror("posix_memalign failed");
      close(fd);
      return -1;
  }

  // Copy the data to the aligned buffer
  snprintf(buf, blockSize, "hello!\n");

  int bytes_write = write(fd, buf, blockSize); // use posix aligned memblock to write
  printf("wrote %d bytes\n", bytes_write);
  if (bytes_write < 0) {
    fprintf(stderr, "write failed: %s\n", strerror(errno));
  }
  printf("closing file: %s\n", myfile);
  close(fd);

  return 0;
}
