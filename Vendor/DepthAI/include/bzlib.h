
/*-------------------------------------------------------------*/
/*--- Public header file for the library.                   ---*/
/*---                                               bzlib.h ---*/
/*-------------------------------------------------------------*/

/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.8 of 13 July 2019
   Copyright (C) 1996-2019 Julian Seward <jseward@acm.org>

   Please read the WARNING, DISCLAIMER and PATENTS sections in the 
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
   ------------------------------------------------------------------ */


#ifndef _BZLIB_H
#define _BZLIB_H

#ifdef __cplusplus
extern "C" {
#endif

#define BZ_RUN               0
#define BZ_FLUSH             1
#define BZ_FINISH            2

#define BZ_OK                0
#define BZ_RUN_OK            1
#define BZ_FLUSH_OK          2
#define BZ_FINISH_OK         3
#define BZ_STREAM_END        4
#define BZ_SEQUENCE_ERROR    (-1)
#define BZ_PARAM_ERROR       (-2)
#define BZ_MEM_ERROR         (-3)
#define BZ_DATA_ERROR        (-4)
#define BZ_DATA_ERROR_MAGIC  (-5)
#define BZ_IO_ERROR          (-6)
#define BZ_UNEXPECTED_EOF    (-7)
#define BZ_OUTBUFF_FULL      (-8)
#define BZ_CONFIG_ERROR      (-9)

typedef 
   struct {
      char *next_in;
      unsigned int avail_in;
      unsigned int total_in_lo32;
      unsigned int total_in_hi32;

      char *next_out;
      unsigned int avail_out;
      unsigned int total_out_lo32;
      unsigned int total_out_hi32;

      void *state;

      void *(*bzalloc)(void *,int,int);
      void (*bzfree)(void *,void *);
      void *opaque;
   } 
   bz_stream;


#ifndef BZ_IMPORT
#define BZ_EXPORT
#endif

/* Need a definitions for FILE */
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#   include <windows.h>
#   ifdef small
      /* windows.h define small to char */
#      undef small
#   endif
#   include <io.h>
#   include <sys/utime.h>
#   define fdopen		_fdopen
#   define isatty		_isatty
#   define setmode		_setmode
#   define utime		_utime
#endif
#ifndef __GNUC__
# define __DLL_IMPORT__ __declspec(dllimport)
# define __DLL_EXPORT__ __declspec(dllexport)
#   else
# define __DLL_IMPORT__ __attribute__((dllimport)) extern
# define __DLL_EXPORT__ __attribute__((dllexport)) extern
#endif

//#if (defined __WIN32__) || (defined _WIN32)
#if 0
# if defined BUILD_BZIP2_DLL  || defined BZ_EXPORT
#  define BZIP2_DLL_IMPEXP __DLL_EXPORT__
# elif defined(BZIP2_STATIC)
#  define BZIP2_DLL_IMPEXP  
# elif defined (USE_BZIP2_DLL) || defined BZ_IMPORT
#  define BZIP2_DLL_IMPEXP __DLL_IMPORT__
# elif defined (USE_BZIP2_STATIC)
#  define BZIP2_DLL_IMPEXP   
# else /* assume USE_BZIP2_DLL */
#  define BZIP2_DLL_IMPEXP __DLL_IMPORT__
#endif
#else /* __WIN32__ */
# define BZIP2_DLL_IMPEXP  
#endif

#define BZ_EXTERN BZIP2_DLL_IMPEXP


/*-- Core (low-level) library functions --*/

BZ_EXTERN int BZ2_bzCompressInit ( 
      bz_stream* strm, 
      int        blockSize100k, 
      int        verbosity, 
      int        workFactor 
   );

BZ_EXTERN int BZ2_bzCompress ( 
      bz_stream* strm, 
      int action 
   );

BZ_EXTERN int BZ2_bzCompressEnd ( 
      bz_stream* strm 
   );

BZ_EXTERN int BZ2_bzDecompressInit ( 
      bz_stream *strm, 
      int       verbosity, 
      int       small
   );

BZ_EXTERN int BZ2_bzDecompress ( 
      bz_stream* strm 
   );

BZ_EXTERN int BZ2_bzDecompressEnd ( 
      bz_stream *strm 
   );



/*-- High(er) level library functions --*/

#define BZ_MAX_UNUSED 5000

typedef void BZFILE;

BZ_EXTERN BZFILE* BZ2_bzReadOpen ( 
      int*  bzerror,   
      FILE* f, 
      int   verbosity, 
      int   small,
      void* unused,    
      int   nUnused 
   );

BZ_EXTERN void BZ2_bzReadClose ( 
      int*    bzerror, 
      BZFILE* b 
   );

BZ_EXTERN void BZ2_bzReadGetUnused ( 
      int*    bzerror, 
      BZFILE* b, 
      void**  unused,  
      int*    nUnused 
   );

BZ_EXTERN int BZ2_bzRead ( 
      int*    bzerror, 
      BZFILE* b, 
      void*   buf, 
      int     len 
   );

BZ_EXTERN BZFILE* BZ2_bzWriteOpen ( 
      int*  bzerror,      
      FILE* f, 
      int   blockSize100k, 
      int   verbosity, 
      int   workFactor 
   );

BZ_EXTERN void BZ2_bzWrite ( 
      int*    bzerror, 
      BZFILE* b, 
      void*   buf, 
      int     len 
   );

BZ_EXTERN void BZ2_bzWriteClose ( 
      int*          bzerror, 
      BZFILE*       b, 
      int           abandon, 
      unsigned int* nbytes_in, 
      unsigned int* nbytes_out 
   );

BZ_EXTERN void BZ2_bzWriteClose64 ( 
      int*          bzerror, 
      BZFILE*       b, 
      int           abandon, 
      unsigned int* nbytes_in_lo32, 
      unsigned int* nbytes_in_hi32, 
      unsigned int* nbytes_out_lo32, 
      unsigned int* nbytes_out_hi32
   );

/*-- Utility functions --*/

BZ_EXTERN int BZ2_bzBuffToBuffCompress ( 
      char*         dest, 
      unsigned int* destLen,
      char*         source, 
      unsigned int  sourceLen,
      int           blockSize100k, 
      int           verbosity, 
      int           workFactor 
   );

BZ_EXTERN int BZ2_bzBuffToBuffDecompress ( 
      char*         dest, 
      unsigned int* destLen,
      char*         source, 
      unsigned int  sourceLen,
      int           small, 
      int           verbosity 
   );


/*--
   Code contributed by Yoshioka Tsuneo (tsuneo@rr.iij4u.or.jp)
   to support better zlib compatibility.
   This code is not _officially_ part of libbzip2 (yet);
   I haven't tested it, documented it, or considered the
   threading-safeness of it.
   If this code breaks, please contact both Yoshioka and me.
--*/

BZ_EXTERN const char * BZ2_bzlibVersion (
      void
   );

BZ_EXTERN BZFILE * BZ2_bzopen (
      const char *path,
      const char *mode
   );

BZ_EXTERN BZFILE * BZ2_bzdopen (
      int        fd,
      const char *mode
   );
         
BZ_EXTERN int BZ2_bzread (
      BZFILE* b, 
      void* buf, 
      int len 
   );

BZ_EXTERN int BZ2_bzwrite (
      BZFILE* b, 
      void*   buf, 
      int     len 
   );

BZ_EXTERN int BZ2_bzflush (
      BZFILE* b
   );

BZ_EXTERN void BZ2_bzclose (
      BZFILE* b
   );

BZ_EXTERN const char * BZ2_bzerror (
      BZFILE *b, 
      int    *errnum
   );

#ifdef __cplusplus
}
#endif

#endif

/*-------------------------------------------------------------*/
/*--- end                                           bzlib.h ---*/
/*-------------------------------------------------------------*/
