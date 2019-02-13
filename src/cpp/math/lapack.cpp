#include <lightspeed/math.hpp>
#include "FCMangle.h" // TODO: Automagically detect

#include <climits>
#include <stdexcept>

#define F_DGEEV  FC_GLOBAL(dgeev,  DGEEV )
#define F_DGESV  FC_GLOBAL(dgesv,  DGESV )
#define F_DGETRF FC_GLOBAL(dgetrf, DGETRF)
#define F_DGETRI FC_GLOBAL(dgetri, DGETRI)
#define F_DPOTRF FC_GLOBAL(dpotrf, DPOTRF)
#define F_DPOTRI FC_GLOBAL(dpotri, DPOTRI)
#define F_DPOTRS FC_GLOBAL(dpotrs, DPOTRS)
#define F_DGESVD FC_GLOBAL(dgesvd, DGESVD)
#define F_DSYEV  FC_GLOBAL(dsyev,  DSYEV )
#define F_DBDSDC FC_GLOBAL(dbdsdc, DBDSDC)
#define F_DBDSQR FC_GLOBAL(dbdsqr, DBDSQR)
#define F_DDISNA FC_GLOBAL(ddisna, DDISNA)
#define F_DGBBRD FC_GLOBAL(dgbbrd, DGBBRD)
#define F_DGBCON FC_GLOBAL(dgbcon, DGBCON)
#define F_DGBEQU FC_GLOBAL(dgbequ, DGBEQU)
#define F_DGBRFS FC_GLOBAL(dgbrfs, DGBRFS)
#define F_DGBSV  FC_GLOBAL(dgbsv,  DGBSV )
#define F_DGBSVX FC_GLOBAL(dgbsvx, DGBSVX)
#define F_DGBTRF FC_GLOBAL(dgbtrf, DGBTRF)
#define F_DGBTRS FC_GLOBAL(dgbtrs, DGBTRS)
#define F_DGEBAK FC_GLOBAL(dgebak, DGEBAK)
#define F_DGEBAL FC_GLOBAL(dgebal, DGEBAL)
#define F_DGEBRD FC_GLOBAL(dgebrd, DGEBRD)
#define F_DGECON FC_GLOBAL(dgecon, DGECON)
#define F_DGEEQU FC_GLOBAL(dgeequ, DGEEQU)
#define F_DGEES  FC_GLOBAL(dgees,  DGEES )
#define F_DGEESX FC_GLOBAL(dgeesx, DGEESX)
#define F_DGEEV  FC_GLOBAL(dgeev,  DGEEV )
#define F_DGEEVX FC_GLOBAL(dgeevx, DGEEVX)
#define F_DGEGS  FC_GLOBAL(dgegs,  DGEGS )
#define F_DGEGV  FC_GLOBAL(dgegv,  DGEGV )
#define F_DGEHRD FC_GLOBAL(dgehrd, DGEHRD)
#define F_DGELQF FC_GLOBAL(dgelqf, DGELQF)
#define F_DGELS  FC_GLOBAL(dgels,  DGELS )
#define F_DGELSD FC_GLOBAL(dgelsd, DGELSD)
#define F_DGELSS FC_GLOBAL(dgelss, DGELSS)
#define F_DGELSX FC_GLOBAL(dgelsx, DGELSX)
#define F_DGELSY FC_GLOBAL(dgelsy, DGELSY)
#define F_DGEQLF FC_GLOBAL(dgeqlf, DGEQLF)
#define F_DGEQP3 FC_GLOBAL(dgeqp3, DGEQP3)
#define F_DGEQPF FC_GLOBAL(dgeqpf, DGEQPF)
#define F_DGEQRF FC_GLOBAL(dgeqrf, DGEQRF)
#define F_DGERFS FC_GLOBAL(dgerfs, DGERFS)
#define F_DGERQF FC_GLOBAL(dgerqf, DGERQF)
#define F_DGESDD FC_GLOBAL(dgesdd, DGESDD)
#define F_DGESV  FC_GLOBAL(dgesv,  DGESV )
#define F_DGESVX FC_GLOBAL(dgesvx, DGESVX)
#define F_DGETRF FC_GLOBAL(dgetrf, DGETRF)
#define F_DGETRI FC_GLOBAL(dgetri, DGETRI)
#define F_DGETRS FC_GLOBAL(dgetrs, DGETRS)
#define F_DGGBAK FC_GLOBAL(dggbak, DGGBAK)
#define F_DGGBAL FC_GLOBAL(dggbal, DGGBAL)
#define F_DGGES  FC_GLOBAL(dgges,  DGGES )
#define F_DGGESX FC_GLOBAL(dggesx, DGGESX)
#define F_DGGEV  FC_GLOBAL(dggev,  DGGEV )
#define F_DGGEVX FC_GLOBAL(dggevx, DGGEVX)
#define F_DGGGLM FC_GLOBAL(dggglm, DGGGLM)
#define F_DGGHRD FC_GLOBAL(dgghrd, DGGHRD)
#define F_DGGLSE FC_GLOBAL(dgglse, DGGLSE)
#define F_DGGQRF FC_GLOBAL(dggqrf, DGGQRF)
#define F_DGGRQF FC_GLOBAL(dggrqf, DGGRQF)
#define F_DGGSVD FC_GLOBAL(dggsvd, DGGSVD)
#define F_DGGSVP FC_GLOBAL(dggsvp, DGGSVP)
#define F_DGTCON FC_GLOBAL(dgtcon, DGTCON)
#define F_DGTRFS FC_GLOBAL(dgtrfs, DGTRFS)
#define F_DGTSV  FC_GLOBAL(dgtsv,  DGTSV )
#define F_DGTSVX FC_GLOBAL(dgtsvx, DGTSVX)
#define F_DGTTRF FC_GLOBAL(dgttrf, DGTTRF)
#define F_DGTTRS FC_GLOBAL(dgttrs, DGTTRS)
#define F_DHGEQZ FC_GLOBAL(dhgeqz, DHGEQZ)
#define F_DHSEIN FC_GLOBAL(dhsein, DHSEIN)
#define F_DHSEQR FC_GLOBAL(dhseqr, DHSEQR)
#define F_DOPGTR FC_GLOBAL(dopgtr, DOPGTR)
#define F_DOPMTR FC_GLOBAL(dopmtr, DOPMTR)
#define F_DORGBR FC_GLOBAL(dorgbr, DORGBR)
#define F_DORGHR FC_GLOBAL(dorghr, DORGHR)
#define F_DORGLQ FC_GLOBAL(dorglq, DORGLQ)
#define F_DORGQL FC_GLOBAL(dorgql, DORGQL)
#define F_DORGQR FC_GLOBAL(dorgqr, DORGQR)
#define F_DORGRQ FC_GLOBAL(dorgrq, DORGRQ)
#define F_DORGTR FC_GLOBAL(dorgtr, DORGTR)
#define F_DORMBR FC_GLOBAL(dormbr, DORMBR)
#define F_DORMHR FC_GLOBAL(dormhr, DORMHR)
#define F_DORMLQ FC_GLOBAL(dormlq, DORMLQ)
#define F_DORMQL FC_GLOBAL(dormql, DORMQL)
#define F_DORMQR FC_GLOBAL(dormqr, DORMQR)
#define F_DORMR3 FC_GLOBAL(dormr3, DORMR3)
#define F_DORMRQ FC_GLOBAL(dormrq, DORMRQ)
#define F_DORMRZ FC_GLOBAL(dormrz, DORMRZ)
#define F_DORMTR FC_GLOBAL(dormtr, DORMTR)
#define F_DPBCON FC_GLOBAL(dpbcon, DPBCON)
#define F_DPBEQU FC_GLOBAL(dpbequ, DPBEQU)
#define F_DPBRFS FC_GLOBAL(dpbrfs, DPBRFS)
#define F_DPBSTF FC_GLOBAL(dpbstf, DPBSTF)
#define F_DPBSV  FC_GLOBAL(dpbsv,  DPBSV )
#define F_DPBSVX FC_GLOBAL(dpbsvx, DPBSVX)
#define F_DPBTRF FC_GLOBAL(dpbtrf, DPBTRF)
#define F_DPBTRS FC_GLOBAL(dpbtrs, DPBTRS)
#define F_DPOCON FC_GLOBAL(dpocon, DPOCON)
#define F_DPOEQU FC_GLOBAL(dpoequ, DPOEQU)
#define F_DPORFS FC_GLOBAL(dporfs, DPORFS)
#define F_DPOSV  FC_GLOBAL(dposv,  DPOSV )
#define F_DPOSVX FC_GLOBAL(dposvx, DPOSVX)
#define F_DPOTRF FC_GLOBAL(dpotrf, DPOTRF)
#define F_DPOTRI FC_GLOBAL(dpotri, DPOTRI)
#define F_DPOTRS FC_GLOBAL(dpotrs, DPOTRS)
#define F_DPPCON FC_GLOBAL(dppcon, DPPCON)
#define F_DPPEQU FC_GLOBAL(dppequ, DPPEQU)
#define F_DPPRFS FC_GLOBAL(dpprfs, DPPRFS)
#define F_DPPSV  FC_GLOBAL(dppsv,  DPPSV )
#define F_DPPSVX FC_GLOBAL(dppsvx, DPPSVX)
#define F_DPPTRF FC_GLOBAL(dpptrf, DPPTRF)
#define F_DPPTRI FC_GLOBAL(dpptri, DPPTRI)
#define F_DPPTRS FC_GLOBAL(dpptrs, DPPTRS)
#define F_DPTCON FC_GLOBAL(dptcon, DPTCON)
#define F_DPTEQR FC_GLOBAL(dpteqr, DPTEQR)
#define F_DPTRFS FC_GLOBAL(dptrfs, DPTRFS)
#define F_DPTSV  FC_GLOBAL(dptsv,  DPTSV )
#define F_DPTSVX FC_GLOBAL(dptsvx, DPTSVX)
#define F_DPTTRF FC_GLOBAL(dpttrf, DPTTRF)
#define F_DPTTRS FC_GLOBAL(dpttrs, DPTTRS)
#define F_DSBEV  FC_GLOBAL(dsbev,  DSBEV )
#define F_DSBEVD FC_GLOBAL(dsbevd, DSBEVD)
#define F_DSBEVX FC_GLOBAL(dsbevx, DSBEVX)
#define F_DSBGST FC_GLOBAL(dsbgst, DSBGST)
#define F_DSBGV  FC_GLOBAL(dsbgv,  DSBGV )
#define F_DSBGVD FC_GLOBAL(dsbgvd, DSBGVD)
#define F_DSBGVX FC_GLOBAL(dsbgvx, DSBGVX)
#define F_DSBTRD FC_GLOBAL(dsbtrd, DSBTRD)
#define F_DSGESV FC_GLOBAL(dsgesv, DSGESV)
#define F_DSPCON FC_GLOBAL(dspcon, DSPCON)
#define F_DSPEV  FC_GLOBAL(dspev,  DSPEV )
#define F_DSPEVD FC_GLOBAL(dspevd, DSPEVD)
#define F_DSPEVX FC_GLOBAL(dspevx, DSPEVX)
#define F_DSPGST FC_GLOBAL(dspgst, DSPGST)
#define F_DSPGV  FC_GLOBAL(dspgv,  DSPGV )
#define F_DSPGVD FC_GLOBAL(dspgvd, DSPGVD)
#define F_DSPGVX FC_GLOBAL(dspgvx, DSPGVX)
#define F_DSPRFS FC_GLOBAL(dsprfs, DSPRFS)
#define F_DSPSV  FC_GLOBAL(dspsv,  DSPSV )
#define F_DSPSVX FC_GLOBAL(dspsvx, DSPSVX)
#define F_DSPTRD FC_GLOBAL(dsptrd, DSPTRD)
#define F_DSPTRF FC_GLOBAL(dsptrf, DSPTRF)
#define F_DSPTRI FC_GLOBAL(dsptri, DSPTRI)
#define F_DSPTRS FC_GLOBAL(dsptrs, DSPTRS)
#define F_DSTEBZ FC_GLOBAL(dstebz, DSTEBZ)
#define F_DSTEDC FC_GLOBAL(dstedc, DSTEDC)
#define F_DSTEGR FC_GLOBAL(dstegr, DSTEGR)
#define F_DSTEIN FC_GLOBAL(dstein, DSTEIN)
#define F_DSTEQR FC_GLOBAL(dsteqr, DSTEQR)
#define F_DSTERF FC_GLOBAL(dsterf, DSTERF)
#define F_DSTEV  FC_GLOBAL(dstev,  DSTEV )
#define F_DSTEVD FC_GLOBAL(dstevd, DSTEVD)
#define F_DSTEVR FC_GLOBAL(dstevr, DSTEVR)
#define F_DSTEVX FC_GLOBAL(dstevx, DSTEVX)
#define F_DSYCON FC_GLOBAL(dsycon, DSYCON)
#define F_DSYEV  FC_GLOBAL(dsyev,  DSYEV )
#define F_DSYEVD FC_GLOBAL(dsyevd, DSYEVD)
#define F_DSYEVR FC_GLOBAL(dsyevr, DSYEVR)
#define F_DSYEVX FC_GLOBAL(dsyevx, DSYEVX)
#define F_DSYGST FC_GLOBAL(dsygst, DSYGST)
#define F_DSYGV  FC_GLOBAL(dsygv,  DSYGV )
#define F_DSYGVD FC_GLOBAL(dsygvd, DSYGVD)
#define F_DSYGVX FC_GLOBAL(dsygvx, DSYGVX)
#define F_DSYRFS FC_GLOBAL(dsyrfs, DSYRFS)
#define F_DSYSV  FC_GLOBAL(dsysv,  DSYSV )
#define F_DSYSVX FC_GLOBAL(dsysvx, DSYSVX)
#define F_DSYTRD FC_GLOBAL(dsytrd, DSYTRD)
#define F_DSYTRF FC_GLOBAL(dsytrf, DSYTRF)
#define F_DSYTRI FC_GLOBAL(dsytri, DSYTRI)
#define F_DSYTRS FC_GLOBAL(dsytrs, DSYTRS)
#define F_DTBCON FC_GLOBAL(dtbcon, DTBCON)
#define F_DTBRFS FC_GLOBAL(dtbrfs, DTBRFS)
#define F_DTBTRS FC_GLOBAL(dtbtrs, DTBTRS)
#define F_DTGEVC FC_GLOBAL(dtgevc, DTGEVC)
#define F_DTGEXC FC_GLOBAL(dtgexc, DTGEXC)
#define F_DTGSEN FC_GLOBAL(dtgsen, DTGSEN)
#define F_DTGSJA FC_GLOBAL(dtgsja, DTGSJA)
#define F_DTGSNA FC_GLOBAL(dtgsna, DTGSNA)
#define F_DTGSYL FC_GLOBAL(dtgsyl, DTGSYL)
#define F_DTPCON FC_GLOBAL(dtpcon, DTPCON)
#define F_DTPRFS FC_GLOBAL(dtprfs, DTPRFS)
#define F_DTPTRI FC_GLOBAL(dtptri, DTPTRI)
#define F_DTPTRS FC_GLOBAL(dtptrs, DTPTRS)
#define F_DTRCON FC_GLOBAL(dtrcon, DTRCON)
#define F_DTREVC FC_GLOBAL(dtrevc, DTREVC)
#define F_DTREXC FC_GLOBAL(dtrexc, DTREXC)
#define F_DTRRFS FC_GLOBAL(dtrrfs, DTRRFS)
#define F_DTRSEN FC_GLOBAL(dtrsen, DTRSEN)
#define F_DTRSNA FC_GLOBAL(dtrsna, DTRSNA)
#define F_DTRSYL FC_GLOBAL(dtrsyl, DTRSYL)
#define F_DTRTRI FC_GLOBAL(dtrtri, DTRTRI)
#define F_DTRTRS FC_GLOBAL(dtrtrs, DTRTRS)
#define F_DTZRQF FC_GLOBAL(dtzrqf, DTZRQF)
#define F_DTZRZF FC_GLOBAL(dtzrzf, DTZRZF)

extern "C" {
extern int F_DBDSDC(char*, char*, int*, double*, double*, double*, int*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DBDSQR(char*, int*, int*, int*, int*, double*, double*, double*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DDISNA(char*, int*, int*, double*, double*,  int*);
extern int F_DGBBRD(char*, int*, int*, int*, int*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DGBCON(char*, int*, int*, int*, double*, int*, int*, double*, double*, double*, int*,  int*);
extern int F_DGBEQU(int*, int*, int*, int*, double*, int*, double*, double*, double*, double*, double*,  int*);
extern int F_DGBRFS(char*, int*, int*, int*, int*, double*, int*, double*, int*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DGBSV(int*, int*, int*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DGBSVX(char*, char*, int*, int*, int*, int*, double*, int*, double*, int*, int*, char*, double*, double*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DGBTRF(int*, int*, int*, int*, double*, int*, int*,  int*);
extern int F_DGBTRS(char*, int*, int*, int*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DGEBAK(char*, char*, int*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DGEBAL(char*, int*, double*, int*, int*, int*, double*,  int*);
extern int F_DGEBRD(int*, int*, double*, int*, double*, double*, double*, double*, double*, int*,  int*);
extern int F_DGECON(char*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DGEEQU(int*, int*, double*, int*, double*, double*, double*, double*, double*,  int*);
extern int F_DGEES(char*, char*, int*, double*, int*, int*, double*, double*, double*, int*, double*, int*,  int*);
extern int F_DGEESX(char*, char*, char*, int*, double*, int*, int*, double*, double*, double*, int*, double*, double*, double*, int*, int*, int*,  int*);
extern int F_DGEEV(char*, char*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGEEVX(char*, char*, char*, char*, int*, double*, int*, double*, double*, double*, int*, double*, int*, int*, int*, double*, double*, double*, double*, double*, int*, int*,  int*);
extern int F_DGEGS(char*, char*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGEGV(char*, char*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGEHRD(int*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DGELQF(int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DGELS(char*, int*, int*, int*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGELSD(int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, int*,  int*);
extern int F_DGELSS(int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DGELSX(int*, int*, int*, double*, int*, double*, int*, int*, double*, int*, double*,  int*);
extern int F_DGELSY(int*, int*, int*, double*, int*, double*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DGEQLF(int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DGEQP3(int*, int*, double*, int*, int*, double*, double*, int*,  int*);
extern int F_DGEQPF(int*, int*, double*, int*, int*, double*, double*,  int*);
extern int F_DGEQRF(int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DGERFS(char*, int*, int*, double*, int*, double*, int*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DGERQF(int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DGESDD(char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*,  int*);
extern int F_DGESV(int*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DGESVX(char*, char*, int*, int*, double*, int*, double*, int*, int*, char*, double*, double*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DGETRF(int*, int*, double*, int*, int*,  int*);
extern int F_DGETRI(int*, double*, int*, int*, double*, int*,  int*);
extern int F_DGETRS(char*, int*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DGGBAK(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DGGBAL(char*, int*, double*, int*, double*, int*, int*, int*, double*, double*, double*,  int*);
extern int F_DGGES(char*, char*, char*, int*, double*, int*, double*, int*, int*, double*, double*, double*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGGESX(char*, char*, char*, char*, int*, double*, int*, double*, int*, int*, double*, double*, double*, double*, int*, double*, int*, double*, double*, double*, int*, int*, int*,  int*);
extern int F_DGGEV(char*, char*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGGEVX(char*, char*, char*, char*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*, double*, int*, int*, int*, double*, double*, double*, double*, double*, double*, double*, int*, int*,  int*);
extern int F_DGGGLM(int*, int*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DGGHRD(char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGGLSE(int*, int*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DGGQRF(int*, int*, int*, double*, int*, double*, double*, int*, double*, double*, int*,  int*);
extern int F_DGGRQF(int*, int*, int*, double*, int*, double*, double*, int*, double*, double*, int*,  int*);
extern int F_DGGSVD(char*, char*, char*, int*, int*, int*, int*, int*, double*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DGGSVP(char*, char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, int*, double*, int*, int*, double*, double*,  int*);
extern int F_DGTCON(char*, int*, double*, double*, double*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DGTRFS(char*, int*, int*, double*, double*, double*, double*, double*, double*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DGTSV(int*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DGTSVX(char*, char*, int*, int*, double*, double*, double*, double*, double*, double*, double*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DGTTRF(int*, double*, double*, double*, double*, int*,  int*);
extern int F_DGTTRS(char*, int*, int*, double*, double*, double*, double*, int*, double*, int*,  int*);
extern int F_DHGEQZ(char*, char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DHSEIN(char*, char*, char*, int*, double*, int*, double*, double*, double*, int*, double*, int*, int*, int*, double*, int*, int*,  int*);
extern int F_DHSEQR(char*, char*, int*, int*, int*, double*, int*, double*, double*, double*, int*, double*, int*,  int*);
extern int F_DOPGTR(char*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DOPMTR(char*, char*, char*, int*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DORGBR(char*, int*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DORGHR(int*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DORGLQ(int*, int*, int*, double*, int*, double*, double*, int*,  int*);
//extern int F_DORGQL(int*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DORGQR(int*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DORGRQ(int*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DORGTR(char*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DORMBR(char*, char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMHR(char*, char*, int*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMLQ(char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMQL(char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMQR(char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMR3(char*, char*, int*, int*, int*, int*, double*, int*, double*, double*, int*, double*,  int*);
extern int F_DORMRQ(char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMRZ(char*, char*, int*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DORMTR(char*, char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*,  int*);
extern int F_DPBCON(char*, int*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DPBEQU(char*, int*, int*, double*, int*, double*, double*, double*,  int*);
extern int F_DPBRFS(char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DPBSTF(char*, int*, int*, double*, int*,  int*);
extern int F_DPBSV(char*, int*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DPBSVX(char*, char*, int*, int*, int*, double*, int*, double*, int*, char*, double*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DPBTRF(char*, int*, int*, double*, int*,  int*);
extern int F_DPBTRS(char*, int*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DPOCON(char*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DPOEQU(int*, double*, int*, double*, double*, double*,  int*);
extern int F_DPORFS(char*, int*, int*, double*, int*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DPOSV(char*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DPOSVX(char*, char*, int*, int*, double*, int*, double*, int*, char*, double*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DPOTRF(char*, int*, double*, int*,  int*);
extern int F_DPOTRI(char*, int*, double*, int*,  int*);
extern int F_DPOTRS(char*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DPPCON(char*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DPPEQU(char*, int*, double*, double*, double*, double*,  int*);
extern int F_DPPRFS(char*, int*, int*, double*, double*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DPPSV(char*, int*, int*, double*, double*, int*,  int*);
extern int F_DPPSVX(char*, char*, int*, int*, double*, double*, char*, double*, double*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DPPTRF(char*, int*, double*,  int*);
extern int F_DPPTRI(char*, int*, double*,  int*);
extern int F_DPPTRS(char*, int*, int*, double*, double*, int*,  int*);
extern int F_DPTCON(int*, double*, double*, double*, double*, double*,  int*);
extern int F_DPTEQR(char*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DPTRFS(int*, int*, double*, double*, double*, double*, double*, int*, double*, int*, double*, double*, double*,  int*);
extern int F_DPTSV(int*, int*, double*, double*, double*, int*,  int*);
extern int F_DPTSVX(char*, int*, int*, double*, double*, double*, double*, double*, int*, double*, int*, double*, double*, double*, double*,  int*);
extern int F_DPTTRF(int*, double*, double*,  int*);
extern int F_DPTTRS(int*, int*, double*, double*, double*, int*,  int*);
extern int F_DSBEV(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*,  int*);
extern int F_DSBEVD(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSBEVX(char*, char*, char*, int*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*,  int*);
extern int F_DSBGST(char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DSBGV(char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*,  int*);
extern int F_DSBGVD(char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSBGVX(char*, char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*,  int*);
extern int F_DSBTRD(char*, char*, int*, int*, double*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DSGESV(int*, int*, double*, int*, int*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DSPCON(char*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DSPEV(char*, char*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DSPEVD(char*, char*, int*, double*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSPEVX(char*, char*, char*, int*, double*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*,  int*);
extern int F_DSPGST(int*, char*, int*, double*, double*,  int*);
extern int F_DSPGV(int*, char*, char*, int*, double*, double*, double*, double*, int*, double*,  int*);
extern int F_DSPGVD(int*, char*, char*, int*, double*, double*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSPGVX(int*, char*, char*, char*, int*, double*, double*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*,  int*);
extern int F_DSPRFS(char*, int*, int*, double*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DSPSV(char*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DSPSVX(char*, char*, int*, int*, double*, double*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DSPTRD(char*, int*, double*, double*, double*, double*,  int*);
extern int F_DSPTRF(char*, int*, double*, int*,  int*);
extern int F_DSPTRI(char*, int*, double*, int*, double*,  int*);
extern int F_DSPTRS(char*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DSTEBZ(char*, char*, int*, double*, double*, int*, int*, double*, double*, double*, int*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DSTEDC(char*, int*, double*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSTEGR(char*, char*, int*, double*, double*, double*, double*, int*, int*, double*, int*, double*, double*, int*, int*, double*, int*, int*, int*,  int*);
extern int F_DSTEIN(int*, double*, double*, int*, double*, int*, int*, double*, int*, double*, int*, int*,  int*);
extern int F_DSTEQR(char*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DSTERF(int*, double*, double*,  int*);
extern int F_DSTEV(char*, int*, double*, double*, double*, int*, double*,  int*);
extern int F_DSTEVD(char*, int*, double*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSTEVR(char*, char*, int*, double*, double*, double*, double*, int*, int*, double*, int*, double*, double*, int*, int*, double*, int*, int*, int*,  int*);
extern int F_DSTEVX(char*, char*, int*, double*, double*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*,  int*);
extern int F_DSYCON(char*, int*, double*, int*, int*, double*, double*, double*, int*,  int*);
extern int F_DSYEV(char*, char*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DSYEVD(char*, char*, int*, double*, int*, double*, double*, int*, int*, int*,  int*);
extern int F_DSYEVR(char*, char*, char*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, int*, double*, int*, int*, int*,  int*);
extern int F_DSYEVX(char*, char*, char*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSYGST(int*, char*, int*, double*, int*, double*, int*,  int*);
extern int F_DSYGV(int*, char*, char*, int*, double*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DSYGVD(int*, char*, char*, int*, double*, int*, double*, int*, double*, double*, int*, int*, int*,  int*);
extern int F_DSYGVX(int*, char*, char*, char*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*,  int*);
extern int F_DSYRFS(char*, int*, int*, double*, int*, double*, int*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DSYSV(char*, int*, int*, double*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DSYSVX(char*, char*, int*, int*, double*, int*, double*, int*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DSYTRD(char*, int*, double*, int*, double*, double*, double*, double*, int*,  int*);
extern int F_DSYTRF(char*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DSYTRI(char*, int*, double*, int*, int*, double*,  int*);
extern int F_DSYTRS(char*, int*, int*, double*, int*, int*, double*, int*,  int*);
extern int F_DTBCON(char*, char*, char*, int*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DTBRFS(char*, char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DTBTRS(char*, char*, char*, int*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DTGEVC(char*, char*, int*, double*, int*, double*, int*, double*, int*, double*, int*, int*, int*, double*,  int*);
extern int F_DTGEXC(int*, double*, int*, double*, int*, double*, int*, double*, int*, int*, int*, double*, int*,  int*);
extern int F_DTGSEN(int*, int*, double*, int*, double*, int*, double*, double*, double*, double*, int*, double*, int*, int*, double*, double*, double*, double*, int*, int*, int*,  int*);
extern int F_DTGSJA(char*, char*, char*, int*, int*, int*, int*, int*, double*, int*, double*, int*, double*, double*, double*, double*, double*, int*, double*, int*, double*, int*, double*, int*,  int*);
extern int F_DTGSNA(char*, char*, int*, double*, int*, double*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, int*,  int*);
extern int F_DTGSYL(char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*, int*,  int*);
extern int F_DTPCON(char*, char*, char*, int*, double*, double*, double*, int*,  int*);
extern int F_DTPRFS(char*, char*, char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DTPTRI(char*, char*, int*, double*,  int*);
extern int F_DTPTRS(char*, char*, char*, int*, int*, double*, double*, int*,  int*);
extern int F_DTRCON(char*, char*, char*, int*, double*, int*, double*, double*, int*,  int*);
extern int F_DTREVC(char*, char*, int*, double*, int*, double*, int*, double*, int*, int*, int*, double*,  int*);
extern int F_DTREXC(char*, int*, double*, int*, double*, int*, int*, int*, double*,  int*);
extern int F_DTRRFS(char*, char*, char*, int*, int*, double*, int*, double*, int*, double*, int*, double*, double*, double*, int*,  int*);
extern int F_DTRSEN(char*, char*, int*, double*, int*, double*, int*, double*, double*, int*, double*, double*, double*, int*, int*, int*,  int*);
extern int F_DTRSNA(char*, char*, int*, double*, int*, double*, int*, double*, int*, double*, double*, int*, int*, double*, int*, int*,  int*);
extern int F_DTRSYL(char*, char*, int*, int*, int*, double*, int*, double*, int*, double*, int*, double*,  int*);
extern int F_DTRTRI(char*, char*, int*, double*, int*,  int*);
extern int F_DTRTRS(char*, char*, char*, int*, int*, double*, int*, double*, int*,  int*);
extern int F_DTZRQF(int*, int*, double*, int*, double*,  int*);
extern int F_DTZRZF(int*, int*, double*, int*, double*, double*, int*,  int*);

extern int F_DGESVD(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);


}

namespace lightspeed {

/**
 *  Purpose
 *  =======
 *
 *  DBDSDC computes the singular value decomposition (SVD) of a real
 *  N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT,
 *  using a divide and conquer method, where S is a diagonal matrix
 *  with non-negative diagonal elements (the singular values of B), and
 *  U and VT are orthogonal matrices of left and right singular vectors,
 *  respectively. DBDSDC can be used to compute all singular values,
 *  and optionally, singular vectors or singular vectors in compact form.
 *
 *  This code makes very mild assumptions about floating point
 *  arithmetic. It will work on machines with a guard digit in
 *  add/subtract, or on those binary machines without guard digits
 *  which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
 *  It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.  See DLASD3 for details.
 *
 *  The code currently calls DLASDQ if singular values only are desired.
 *  However, it can be slightly modified to compute singular values
 *  using the divide and conquer method.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  B is upper bidiagonal.
 *          = 'L':  B is lower bidiagonal.
 *
 *  COMPQ   (input) CHARACTER*1
 *          Specifies whether singular vectors are to be computed
 *          as follows:
 *          = 'N':  Compute singular values only;
 *          = 'P':  Compute singular values and compute singular
 *                  vectors in compact form;
 *          = 'I':  Compute singular values and singular vectors.
 *
 *  N       (input) INTEGER
 *          The order of the matrix B.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the bidiagonal matrix B.
 *          On exit, if INFO=0, the singular values of B.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the elements of E contain the offdiagonal
 *          elements of the bidiagonal matrix whose SVD is desired.
 *          On exit, E has been destroyed.
 *
 *  U       (output) DOUBLE PRECISION array, dimension (LDU,N)
 *          If  COMPQ = 'I', then:
 *             On exit, if INFO = 0, U contains the left singular vectors
 *             of the bidiagonal matrix.
 *          For other values of COMPQ, U is not referenced.
 *
 *  LDU     (input) INTEGER
 *          The leading dimension of the array U.  LDU >= 1.
 *          If singular vectors are desired, then LDU >= max( 1, N ).
 *
 *  VT      (output) DOUBLE PRECISION array, dimension (LDVT,N)
 *          If  COMPQ = 'I', then:
 *             On exit, if INFO = 0, VT' contains the right singular
 *             vectors of the bidiagonal matrix.
 *          For other values of COMPQ, VT is not referenced.
 *
 *  LDVT    (input) INTEGER
 *          The leading dimension of the array VT.  LDVT >= 1.
 *          If singular vectors are desired, then LDVT >= max( 1, N ).
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ)
 *          If  COMPQ = 'P', then:
 *             On exit, if INFO = 0, Q and IQ contain the left
 *             and right singular vectors in a compact form,
 *             requiring O(N log N) space instead of 2*N**2.
 *             In particular, Q contains all the DOUBLE PRECISION data in
 *             LDQ >= N*(11 + 2*SMLSIZ + 8*INT(LOG_2(N/(SMLSIZ+1))))
 *             words of memory, where SMLSIZ is returned by ILAENV and
 *             is equal to the maximum size of the subproblems at the
 *             bottom of the computation tree (usually about 25).
 *          For other values of COMPQ, Q is not referenced.
 *
 *  IQ      (output) INTEGER array, dimension (LDIQ)
 *          If  COMPQ = 'P', then:
 *             On exit, if INFO = 0, Q and IQ contain the left
 *             and right singular vectors in a compact form,
 *             requiring O(N log N) space instead of 2*N**2.
 *             In particular, IQ contains all INTEGER data in
 *             LDIQ >= N*(3 + 3*INT(LOG_2(N/(SMLSIZ+1))))
 *             words of memory, where SMLSIZ is returned by ILAENV and
 *             is equal to the maximum size of the subproblems at the
 *             bottom of the computation tree (usually about 25).
 *          For other values of COMPQ, IQ is not referenced.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          If COMPQ = 'N' then LWORK >= (4 * N).
 *          If COMPQ = 'P' then LWORK >= (6 * N).
 *          If COMPQ = 'I' then LWORK >= (3 * N**2 + 4 * N).
 *
 *  IWORK   (workspace) INTEGER array, dimension (8*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  The algorithm failed to compute a singular value.
 *                The update process of divide and conquer failed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Ming Gu and Huan Ren, Computer Science Division, University of
 *     California at Berkeley, USA
 *
 *  =====================================================================
 *  Changed dimension statement in comment describing E from (N) to
 *  (N-1).  Sven, 17 Feb 05.
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DBDSDC(char uplo, char compq, int n, double* d, double* e, double* u, int ldu, double* vt, int ldvt, double* q, int* iq, double* work, int* iwork)
{
    int info;
    ::F_DBDSDC(&uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DBDSQR computes the singular values and, optionally, the right and/or
 *  left singular vectors from the singular value decomposition (SVD) of
 *  a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
 *  zero-shift QR algorithm.  The SVD of B has the form
 *
 *     B = Q * S * P**T
 *
 *  where S is the diagonal matrix of singular values, Q is an orthogonal
 *  matrix of left singular vectors, and P is an orthogonal matrix of
 *  right singular vectors.  If left singular vectors are requested, this
 *  subroutine actually returns U*Q instead of Q, and, if right singular
 *  vectors are requested, this subroutine returns P**T*VT instead of
 *  P**T, for given real input matrices U and VT.  When U and VT are the
 *  orthogonal matrices that reduce a general matrix A to bidiagonal
 *  form:  A = U*B*VT, as computed by DGEBRD, then
 *
 *     A = (U*Q) * S * (P**T*VT)
 *
 *  is the SVD of A.  Optionally, the subroutine may also compute Q**T*C
 *  for a given real input matrix C.
 *
 *  See "Computing  Small Singular Values of Bidiagonal Matrices With
 *  Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
 *  LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
 *  no. 5, pp. 873-912, Sept 1990) and
 *  "Accurate singular values and differential qd algorithms," by
 *  B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
 *  Department, University of California at Berkeley, July 1992
 *  for a detailed description of the algorithm.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  B is upper bidiagonal;
 *          = 'L':  B is lower bidiagonal.
 *
 *  N       (input) INTEGER
 *          The order of the matrix B.  N >= 0.
 *
 *  NCVT    (input) INTEGER
 *          The number of columns of the matrix VT. NCVT >= 0.
 *
 *  NRU     (input) INTEGER
 *          The number of rows of the matrix U. NRU >= 0.
 *
 *  NCC     (input) INTEGER
 *          The number of columns of the matrix C. NCC >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the bidiagonal matrix B.
 *          On exit, if INFO=0, the singular values of B in decreasing
 *          order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the N-1 offdiagonal elements of the bidiagonal
 *          matrix B.
 *          On exit, if INFO = 0, E is destroyed; if INFO > 0, D and E
 *          will contain the diagonal and superdiagonal elements of a
 *          bidiagonal matrix orthogonally equivalent to the one given
 *          as input.
 *
 *  VT      (input/output) DOUBLE PRECISION array, dimension (LDVT, NCVT)
 *          On entry, an N-by-NCVT matrix VT.
 *          On exit, VT is overwritten by P**T * VT.
 *          Not referenced if NCVT = 0.
 *
 *  LDVT    (input) INTEGER
 *          The leading dimension of the array VT.
 *          LDVT >= max(1,N) if NCVT > 0; LDVT >= 1 if NCVT = 0.
 *
 *  U       (input/output) DOUBLE PRECISION array, dimension (LDU, N)
 *          On entry, an NRU-by-N matrix U.
 *          On exit, U is overwritten by U * Q.
 *          Not referenced if NRU = 0.
 *
 *  LDU     (input) INTEGER
 *          The leading dimension of the array U.  LDU >= max(1,NRU).
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC, NCC)
 *          On entry, an N-by-NCC matrix C.
 *          On exit, C is overwritten by Q**T * C.
 *          Not referenced if NCC = 0.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C.
 *          LDC >= max(1,N) if NCC > 0; LDC >=1 if NCC = 0.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (4*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  If INFO = -i, the i-th argument had an illegal value
 *          > 0:
 *             if NCVT = NRU = NCC = 0,
 *                = 1, a split was marked by a positive value in E
 *                = 2, current block of Z not diagonalized after 30*N
 *                     iterations (in inner while loop)
 *                = 3, termination criterion of outer while loop not met
 *                     (program created more than N unreduced blocks)
 *             else NCVT = NRU = NCC = 0,
 *                   the algorithm did not converge; D and E contain the
 *                   elements of a bidiagonal matrix which is orthogonally
 *                   similar to the input matrix B;  if INFO = i, i
 *                   elements of E have not converged to zero.
 *
 *  Internal Parameters
 *  ===================
 *
 *  TOLMUL  DOUBLE PRECISION, default = max(10,min(100,EPS**(-1/8)))
 *          TOLMUL controls the convergence criterion of the QR loop.
 *          If it is positive, TOLMUL*EPS is the desired relative
 *             precision in the computed singular values.
 *          If it is negative, abs(TOLMUL*EPS*sigma_max) is the
 *             desired absolute accuracy in the computed singular
 *             values (corresponds to relative accuracy
 *             abs(TOLMUL*EPS) in the largest singular value.
 *          abs(TOLMUL) should be between 1 and 1/EPS, and preferably
 *             between 10 (for fast convergence) and .1/EPS
 *             (for there to be some accuracy in the results).
 *          Default is to lose at either one eighth or 2 of the
 *             available decimal digits in each computed singular value
 *             (whichever is smaller).
 *
 *  MAXITR  INTEGER, default = 6
 *          MAXITR controls the maximum number of passes of the
 *          algorithm through its inner loop. The algorithms stops
 *          (and so fails to converge) if the number of passes
 *          through the inner loop exceeds MAXITR*N**2.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DBDSQR(char uplo, int n, int ncvt, int nru, int ncc, double* d, double* e, double* vt, int ldvt, double* u, int ldu, double* c, int ldc, double* work)
{
    int info;
    ::F_DBDSQR(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DDISNA computes the reciprocal condition numbers for the eigenvectors
 *  of a real symmetric or complex Hermitian matrix or for the left or
 *  right singular vectors of a general m-by-n matrix. The reciprocal
 *  condition number is the 'gap' between the corresponding eigenvalue or
 *  singular value and the nearest other one.
 *
 *  The bound on the error, measured by angle in radians, in the I-th
 *  computed vector is given by
 *
 *         DLAMCH( 'E' ) * ( ANORM / SEP( I ) )
 *
 *  where ANORM = 2-norm(A) = max( abs( D(j) ) ).  SEP(I) is not allowed
 *  to be smaller than DLAMCH( 'E' )*ANORM in order to limit the size of
 *  the error bound.
 *
 *  DDISNA may also be used to compute error bounds for eigenvectors of
 *  the generalized symmetric definite eigenproblem.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies for which problem the reciprocal condition numbers
 *          should be computed:
 *          = 'E':  the eigenvectors of a symmetric/Hermitian matrix;
 *          = 'L':  the left singular vectors of a general matrix;
 *          = 'R':  the right singular vectors of a general matrix.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix. M >= 0.
 *
 *  N       (input) INTEGER
 *          If JOB = 'L' or 'R', the number of columns of the matrix,
 *          in which case N >= 0. Ignored if JOB = 'E'.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (M) if JOB = 'E'
 *                              dimension (min(M,N)) if JOB = 'L' or 'R'
 *          The eigenvalues (if JOB = 'E') or singular values (if JOB =
 *          'L' or 'R') of the matrix, in either increasing or decreasing
 *          order. If singular values, they must be non-negative.
 *
 *  SEP     (output) DOUBLE PRECISION array, dimension (M) if JOB = 'E'
 *                               dimension (min(M,N)) if JOB = 'L' or 'R'
 *          The reciprocal condition numbers of the vectors.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DDISNA(char job, int m, int n, double* d, double* sep)
{
    int info;
    ::F_DDISNA(&job, &m, &n, d, sep, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBBRD reduces a real general m-by-n band matrix A to upper
 *  bidiagonal form B by an orthogonal transformation: Q' * A * P = B.
 *
 *  The routine computes B, and optionally forms Q or P', or computes
 *  Q'*C for a given matrix C.
 *
 *  Arguments
 *  =========
 *
 *  VECT    (input) CHARACTER*1
 *          Specifies whether or not the matrices Q and P' are to be
 *          formed.
 *          = 'N': do not form Q or P';
 *          = 'Q': form Q only;
 *          = 'P': form P' only;
 *          = 'B': form both.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  NCC     (input) INTEGER
 *          The number of columns of the matrix C.  NCC >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals of the matrix A. KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals of the matrix A. KU >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the m-by-n band matrix A, stored in rows 1 to
 *          KL+KU+1. The j-th column of A is stored in the j-th column of
 *          the array AB as follows:
 *          AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl).
 *          On exit, A is overwritten by values generated during the
 *          reduction.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array A. LDAB >= KL+KU+1.
 *
 *  D       (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The diagonal elements of the bidiagonal matrix B.
 *
 *  E       (output) DOUBLE PRECISION array, dimension (min(M,N)-1)
 *          The superdiagonal elements of the bidiagonal matrix B.
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ,M)
 *          If VECT = 'Q' or 'B', the m-by-m orthogonal matrix Q.
 *          If VECT = 'N' or 'P', the array Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.
 *          LDQ >= max(1,M) if VECT = 'Q' or 'B'; LDQ >= 1 otherwise.
 *
 *  PT      (output) DOUBLE PRECISION array, dimension (LDPT,N)
 *          If VECT = 'P' or 'B', the n-by-n orthogonal matrix P'.
 *          If VECT = 'N' or 'Q', the array PT is not referenced.
 *
 *  LDPT    (input) INTEGER
 *          The leading dimension of the array PT.
 *          LDPT >= max(1,N) if VECT = 'P' or 'B'; LDPT >= 1 otherwise.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,NCC)
 *          On entry, an m-by-ncc matrix C.
 *          On exit, C is overwritten by Q'*C.
 *          C is not referenced if NCC = 0.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C.
 *          LDC >= max(1,M) if NCC > 0; LDC >= 1 if NCC = 0.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*max(M,N))
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBBRD(char vect, int m, int n, int ncc, int kl, int ku, double* ab, int ldab, double* d, double* e, double* q, int ldq, double* pt, int ldpt, double* c, int ldc, double* work)
{
    int info;
    ::F_DGBBRD(&vect, &m, &n, &ncc, &kl, &ku, ab, &ldab, d, e, q, &ldq, pt, &ldpt, c, &ldc, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBCON estimates the reciprocal of the condition number of a real
 *  general band matrix A, in either the 1-norm or the infinity-norm,
 *  using the LU factorization computed by DGBTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as
 *     RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 *  Arguments
 *  =========
 *
 *  NORM    (input) CHARACTER*1
 *          Specifies whether the 1-norm condition number or the
 *          infinity-norm condition number is required:
 *          = '1' or 'O':  1-norm;
 *          = 'I':         Infinity-norm.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          Details of the LU factorization of the band matrix A, as
 *          computed by DGBTRF.  U is stored as an upper triangular band
 *          matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, and
 *          the multipliers used during the factorization are stored in
 *          rows KL+KU+2 to 2*KL+KU+1.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices; for 1 <= i <= N, row i of the matrix was
 *          interchanged with row IPIV(i).
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          If NORM = '1' or 'O', the 1-norm of the original matrix A.
 *          If NORM = 'I', the infinity-norm of the original matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(norm(A) * norm(inv(A))).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBCON(char norm, int n, int kl, int ku, double* ab, int ldab, int* ipiv, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DGBCON(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBEQU computes row and column scalings intended to equilibrate an
 *  M-by-N band matrix A and reduce its condition number.  R returns the
 *  row scale factors and C the column scale factors, chosen to try to
 *  make the largest element in each row and column of the matrix B with
 *  elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.
 *
 *  R(i) and C(j) are restricted to be between SMLNUM = smallest safe
 *  number and BIGNUM = largest safe number.  Use of these scaling
 *  factors is not guaranteed to reduce the condition number of A but
 *  works well in practice.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The band matrix A, stored in rows 1 to KL+KU+1.  The j-th
 *          column of A is stored in the j-th column of the array AB as
 *          follows:
 *          AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KL+KU+1.
 *
 *  R       (output) DOUBLE PRECISION array, dimension (M)
 *          If INFO = 0, or INFO > M, R contains the row scale factors
 *          for A.
 *
 *  C       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, C contains the column scale factors for A.
 *
 *  ROWCND  (output) DOUBLE PRECISION
 *          If INFO = 0 or INFO > M, ROWCND contains the ratio of the
 *          smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and
 *          AMAX is neither too large nor too small, it is not worth
 *          scaling by R.
 *
 *  COLCND  (output) DOUBLE PRECISION
 *          If INFO = 0, COLCND contains the ratio of the smallest
 *          C(i) to the largest C(i).  If COLCND >= 0.1, it is not
 *          worth scaling by C.
 *
 *  AMAX    (output) DOUBLE PRECISION
 *          Absolute value of largest matrix element.  If AMAX is very
 *          close to overflow or very close to underflow, the matrix
 *          should be scaled.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= M:  the i-th row of A is exactly zero
 *                >  M:  the (i-M)-th column of A is exactly zero
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBEQU(int m, int n, int kl, int ku, double* ab, int ldab, double* r, double* c, double* rowcnd, double* colcnd, double* amax)
{
    int info;
    ::F_DGBEQU(&m, &n, &kl, &ku, ab, &ldab, r, c, rowcnd, colcnd, amax, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is banded, and provides
 *  error bounds and backward error estimates for the solution.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B     (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The original band matrix A, stored in rows 1 to KL+KU+1.
 *          The j-th column of A is stored in the j-th column of the
 *          array AB as follows:
 *          AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(n,j+kl).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KL+KU+1.
 *
 *  AFB     (input) DOUBLE PRECISION array, dimension (LDAFB,N)
 *          Details of the LU factorization of the band matrix A, as
 *          computed by DGBTRF.  U is stored as an upper triangular band
 *          matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, and
 *          the multipliers used during the factorization are stored in
 *          rows KL+KU+2 to 2*KL+KU+1.
 *
 *  LDAFB   (input) INTEGER
 *          The leading dimension of the array AFB.  LDAFB >= 2*KL*KU+1.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices from DGBTRF; for 1<=i<=N, row i of the
 *          matrix was interchanged with row IPIV(i).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DGBTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBRFS(char trans, int n, int kl, int ku, int nrhs, double* ab, int ldab, double* afb, int ldafb, int* ipiv, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DGBRFS(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBSV computes the solution to a real system of linear equations
 *  A * X = B, where A is a band matrix of order N with KL subdiagonals
 *  and KU superdiagonals, and X and B are N-by-NRHS matrices.
 *
 *  The LU decomposition with partial pivoting and row interchanges is
 *  used to factor A as A = L * U, where L is a product of permutation
 *  and unit lower triangular matrices with KL subdiagonals, and U is
 *  upper triangular with KL+KU superdiagonals.  The factored form of A
 *  is then used to solve the system of equations A * X = B.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the matrix A in band storage, in rows KL+1 to
 *          2*KL+KU+1; rows 1 to KL of the array need not be set.
 *          The j-th column of A is stored in the j-th column of the
 *          array AB as follows:
 *          AB(KL+KU+1+i-j,j) = A(i,j) for max(1,j-KU)<=i<=min(N,j+KL)
 *          On exit, details of the factorization: U is stored as an
 *          upper triangular band matrix with KL+KU superdiagonals in
 *          rows 1 to KL+KU+1, and the multipliers used during the
 *          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
 *          See below for further details.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          The pivot indices that define the permutation matrix P;
 *          row i of the matrix was interchanged with row IPIV(i).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, U(i,i) is exactly zero.  The factorization
 *                has been completed, but the factor U is exactly
 *                singular, and the solution has not been computed.
 *
 *  Further Details
 *  ===============
 *
 *  The band storage scheme is illustrated by the following example, when
 *  M = N = 6, KL = 2, KU = 1:
 *
 *  On entry:                       On exit:
 *
 *      *    *    *    +    +    +       *    *    *   u14  u25  u36
 *      *    *    +    +    +    +       *    *   u13  u24  u35  u46
 *      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
 *     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
 *     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
 *     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *
 *
 *  Array elements marked * are not used by the routine; elements marked
 *  + need not be set on entry, but are required by the routine to store
 *  elements of U because of fill-in resulting from the row interchanges.
 *
 *  =====================================================================
 *
 *     .. External Subroutines ..
 **/
int C_DGBSV(int n, int kl, int ku, int nrhs, double* ab, int ldab, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DGBSV(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBSVX uses the LU factorization to compute the solution to a real
 *  system of linear equations A * X = B, A**T * X = B, or A**H * X = B,
 *  where A is a band matrix of order N with KL subdiagonals and KU
 *  superdiagonals, and X and B are N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed by this subroutine:
 *
 *  1. If FACT = 'E', real scaling factors are computed to equilibrate
 *     the system:
 *        TRANS = 'N':  diag(R)*A*diag(C)     *inv(diag(C))*X = diag(R)*B
 *        TRANS = 'T': (diag(R)*A*diag(C))**T *inv(diag(R))*X = diag(C)*B
 *        TRANS = 'C': (diag(R)*A*diag(C))**H *inv(diag(R))*X = diag(C)*B
 *     Whether or not the system will be equilibrated depends on the
 *     scaling of the matrix A, but if equilibration is used, A is
 *     overwritten by diag(R)*A*diag(C) and B by diag(R)*B (if TRANS='N')
 *     or diag(C)*B (if TRANS = 'T' or 'C').
 *
 *  2. If FACT = 'N' or 'E', the LU decomposition is used to factor the
 *     matrix A (after equilibration if FACT = 'E') as
 *        A = L * U,
 *     where L is a product of permutation and unit lower triangular
 *     matrices with KL subdiagonals, and U is upper triangular with
 *     KL+KU superdiagonals.
 *
 *  3. If some U(i,i)=0, so that U is exactly singular, then the routine
 *     returns with INFO = i. Otherwise, the factored form of A is used
 *     to estimate the condition number of the matrix A.  If the
 *     reciprocal of the condition number is less than machine precision,
 *  C++ Return value: INFO    (output) INTEGER
 *     to solve for X and compute error bounds as described below.
 *
 *  4. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  5. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  6. If equilibration was used, the matrix X is premultiplied by
 *     diag(C) (if TRANS = 'N') or diag(R) (if TRANS = 'T' or 'C') so
 *     that it solves the original system before equilibration.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of the matrix A is
 *          supplied on entry, and if not, whether the matrix A should be
 *          equilibrated before it is factored.
 *          = 'F':  On entry, AFB and IPIV contain the factored form of
 *                  A.  If EQUED is not 'N', the matrix A has been
 *                  equilibrated with scaling factors given by R and C.
 *                  AB, AFB, and IPIV are not modified.
 *          = 'N':  The matrix A will be copied to AFB and factored.
 *          = 'E':  The matrix A will be equilibrated if necessary, then
 *                  copied to AFB and factored.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations.
 *          = 'N':  A * X = B     (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Transpose)
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the matrix A in band storage, in rows 1 to KL+KU+1.
 *          The j-th column of A is stored in the j-th column of the
 *          array AB as follows:
 *          AB(KU+1+i-j,j) = A(i,j) for max(1,j-KU)<=i<=min(N,j+kl)
 *
 *          If FACT = 'F' and EQUED is not 'N', then A must have been
 *          equilibrated by the scaling factors in R and/or C.  AB is not
 *          modified if FACT = 'F' or 'N', or if FACT = 'E' and
 *          EQUED = 'N' on exit.
 *
 *          On exit, if EQUED .ne. 'N', A is scaled as follows:
 *          EQUED = 'R':  A := diag(R) * A
 *          EQUED = 'C':  A := A * diag(C)
 *          EQUED = 'B':  A := diag(R) * A * diag(C).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KL+KU+1.
 *
 *  AFB     (input or output) DOUBLE PRECISION array, dimension (LDAFB,N)
 *          If FACT = 'F', then AFB is an input argument and on entry
 *          contains details of the LU factorization of the band matrix
 *          A, as computed by DGBTRF.  U is stored as an upper triangular
 *          band matrix with KL+KU superdiagonals in rows 1 to KL+KU+1,
 *          and the multipliers used during the factorization are stored
 *          in rows KL+KU+2 to 2*KL+KU+1.  If EQUED .ne. 'N', then AFB is
 *          the factored form of the equilibrated matrix A.
 *
 *          If FACT = 'N', then AFB is an output argument and on exit
 *          returns details of the LU factorization of A.
 *
 *          If FACT = 'E', then AFB is an output argument and on exit
 *          returns details of the LU factorization of the equilibrated
 *          matrix A (see the description of AB for the form of the
 *          equilibrated matrix).
 *
 *  LDAFB   (input) INTEGER
 *          The leading dimension of the array AFB.  LDAFB >= 2*KL+KU+1.
 *
 *  IPIV    (input or output) INTEGER array, dimension (N)
 *          If FACT = 'F', then IPIV is an input argument and on entry
 *          contains the pivot indices from the factorization A = L*U
 *          as computed by DGBTRF; row i of the matrix was interchanged
 *          with row IPIV(i).
 *
 *          If FACT = 'N', then IPIV is an output argument and on exit
 *          contains the pivot indices from the factorization A = L*U
 *          of the original matrix A.
 *
 *          If FACT = 'E', then IPIV is an output argument and on exit
 *          contains the pivot indices from the factorization A = L*U
 *          of the equilibrated matrix A.
 *
 *  EQUED   (input or output) CHARACTER*1
 *          Specifies the form of equilibration that was done.
 *          = 'N':  No equilibration (always true if FACT = 'N').
 *          = 'R':  Row equilibration, i.e., A has been premultiplied by
 *                  diag(R).
 *          = 'C':  Column equilibration, i.e., A has been postmultiplied
 *                  by diag(C).
 *          = 'B':  Both row and column equilibration, i.e., A has been
 *                  replaced by diag(R) * A * diag(C).
 *          EQUED is an input argument if FACT = 'F'; otherwise, it is an
 *          output argument.
 *
 *  R       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The row scale factors for A.  If EQUED = 'R' or 'B', A is
 *          multiplied on the left by diag(R); if EQUED = 'N' or 'C', R
 *          is not accessed.  R is an input argument if FACT = 'F';
 *          otherwise, R is an output argument.  If FACT = 'F' and
 *          EQUED = 'R' or 'B', each element of R must be positive.
 *
 *  C       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The column scale factors for A.  If EQUED = 'C' or 'B', A is
 *          multiplied on the right by diag(C); if EQUED = 'N' or 'R', C
 *          is not accessed.  C is an input argument if FACT = 'F';
 *          otherwise, C is an output argument.  If FACT = 'F' and
 *          EQUED = 'C' or 'B', each element of C must be positive.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit,
 *          if EQUED = 'N', B is not modified;
 *          if TRANS = 'N' and EQUED = 'R' or 'B', B is overwritten by
 *          diag(R)*B;
 *          if TRANS = 'T' or 'C' and EQUED = 'C' or 'B', B is
 *          overwritten by diag(C)*B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X
 *          to the original system of equations.  Note that A and B are
 *          modified on exit if EQUED .ne. 'N', and the solution to the
 *          equilibrated system is inv(diag(C))*X if TRANS = 'N' and
 *          EQUED = 'C' or 'B', or inv(diag(R))*X if TRANS = 'T' or 'C'
 *          and EQUED = 'R' or 'B'.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A after equilibration (if done).  If RCOND is less than the
 *          machine precision (in particular, if RCOND = 0), the matrix
 *          is singular to working precision.  This condition is
 *          indicated by a return code of INFO > 0.
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (3*N)
 *          On exit, WORK(1) contains the reciprocal pivot growth
 *          factor norm(A)/norm(U). The "max absolute element" norm is
 *          used. If WORK(1) is much less than 1, then the stability
 *          of the LU factorization of the (equilibrated) matrix A
 *          could be poor. This also means that the solution X, condition
 *          estimator RCOND, and forward error bound FERR could be
 *          unreliable. If factorization fails with 0<INFO<=N, then
 *          WORK(1) contains the reciprocal pivot growth factor for the
 *          leading INFO columns of A.
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  U(i,i) is exactly zero.  The factorization
 *                       has been completed, but the factor U is exactly
 *                       singular, so the solution and error bounds
 *                       could not be computed. RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBSVX(char fact, char trans, int n, int kl, int ku, int nrhs, double* ab, int ldab, double* afb, int ldafb, int* ipiv, char equed, double* r, double* c, double* b, int ldb, double* x, int ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DGBSVX(&fact, &trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv, &equed, r, c, b, &ldb, x, &ldx, rcond, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBTRF computes an LU factorization of a real m-by-n band matrix A
 *  using partial pivoting with row interchanges.
 *
 *  This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the matrix A in band storage, in rows KL+1 to
 *          2*KL+KU+1; rows 1 to KL of the array need not be set.
 *          The j-th column of A is stored in the j-th column of the
 *          array AB as follows:
 *          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)
 *
 *          On exit, details of the factorization: U is stored as an
 *          upper triangular band matrix with KL+KU superdiagonals in
 *          rows 1 to KL+KU+1, and the multipliers used during the
 *          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
 *          See below for further details.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.
 *
 *  IPIV    (output) INTEGER array, dimension (min(M,N))
 *          The pivot indices; for 1 <= i <= min(M,N), row i of the
 *          matrix was interchanged with row IPIV(i).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = +i, U(i,i) is exactly zero. The factorization
 *               has been completed, but the factor U is exactly
 *               singular, and division by zero will occur if it is used
 *               to solve a system of equations.
 *
 *  Further Details
 *  ===============
 *
 *  The band storage scheme is illustrated by the following example, when
 *  M = N = 6, KL = 2, KU = 1:
 *
 *  On entry:                       On exit:
 *
 *      *    *    *    +    +    +       *    *    *   u14  u25  u36
 *      *    *    +    +    +    +       *    *   u13  u24  u35  u46
 *      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
 *     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
 *     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
 *     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *
 *
 *  Array elements marked * are not used by the routine; elements marked
 *  + need not be set on entry, but are required by the routine to store
 *  elements of U because of fill-in resulting from the row interchanges.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBTRF(int m, int n, int kl, int ku, double* ab, int ldab, int* ipiv)
{
    int info;
    ::F_DGBTRF(&m, &n, &kl, &ku, ab, &ldab, ipiv, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBTRS solves a system of linear equations
 *     A * X = B  or  A' * X = B
 *  with a general band matrix A using the LU factorization computed
 *  by DGBTRF.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations.
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A'* X = B  (Transpose)
 *          = 'C':  A'* X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KL      (input) INTEGER
 *          The number of subdiagonals within the band of A.  KL >= 0.
 *
 *  KU      (input) INTEGER
 *          The number of superdiagonals within the band of A.  KU >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          Details of the LU factorization of the band matrix A, as
 *          computed by DGBTRF.  U is stored as an upper triangular band
 *          matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, and
 *          the multipliers used during the factorization are stored in
 *          rows KL+KU+2 to 2*KL+KU+1.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices; for 1 <= i <= N, row i of the matrix was
 *          interchanged with row IPIV(i).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGBTRS(char trans, int n, int kl, int ku, int nrhs, double* ab, int ldab, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DGBTRS(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEBAK forms the right or left eigenvectors of a real general matrix
 *  by backward transformation on the computed eigenvectors of the
 *  balanced matrix output by DGEBAL.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies the type of backward transformation required:
 *          = 'N', do nothing, return immediately;
 *          = 'P', do backward transformation for permutation only;
 *          = 'S', do backward transformation for scaling only;
 *          = 'B', do backward transformations for both permutation and
 *                 scaling.
 *          JOB must be the same as the argument JOB supplied to DGEBAL.
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'R':  V contains right eigenvectors;
 *          = 'L':  V contains left eigenvectors.
 *
 *  N       (input) INTEGER
 *          The number of rows of the matrix V.  N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          The integers ILO and IHI determined by DGEBAL.
 *          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
 *
 *  SCALE   (input) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutation and scaling factors, as returned
 *          by DGEBAL.
 *
 *  M       (input) INTEGER
 *          The number of columns of the matrix V.  M >= 0.
 *
 *  V       (input/output) DOUBLE PRECISION array, dimension (LDV,M)
 *          On entry, the matrix of right or left eigenvectors to be
 *          transformed, as returned by DHSEIN or DTREVC.
 *          On exit, V is overwritten by the transformed eigenvectors.
 *
 *  LDV     (input) INTEGER
 *          The leading dimension of the array V. LDV >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEBAK(char job, char side, int n, int ilo, int ihi, double* scale, int m, double* v, int ldv)
{
    int info;
    ::F_DGEBAK(&job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEBAL balances a general real matrix A.  This involves, first,
 *  permuting A by a similarity transformation to isolate eigenvalues
 *  in the first 1 to ILO-1 and last IHI+1 to N elements on the
 *  diagonal; and second, applying a diagonal similarity transformation
 *  to rows and columns ILO to IHI to make the rows and columns as
 *  close in norm as possible.  Both steps are optional.
 *
 *  Balancing may reduce the 1-norm of the matrix, and improve the
 *  accuracy of the computed eigenvalues and/or eigenvectors.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies the operations to be performed on A:
 *          = 'N':  none:  simply set ILO = 1, IHI = N, SCALE(I) = 1.0
 *                  for i = 1,...,N;
 *          = 'P':  permute only;
 *          = 'S':  scale only;
 *          = 'B':  both permute and scale.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the input matrix A.
 *          On exit,  A is overwritten by the balanced matrix.
 *          If JOB = 'N', A is not referenced.
 *          See Further Details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  ILO     (output) INTEGER
 *  IHI     (output) INTEGER
 *          ILO and IHI are set to integers such that on exit
 *          A(i,j) = 0 if i > j and j = 1,...,ILO-1 or I = IHI+1,...,N.
 *          If JOB = 'N' or 'S', ILO = 1 and IHI = N.
 *
 *  SCALE   (output) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and scaling factors applied to
 *          A.  If P(j) is the index of the row and column interchanged
 *          with row and column j and D(j) is the scaling factor
 *          applied to row and column j, then
 *          SCALE(j) = P(j)    for j = 1,...,ILO-1
 *                   = D(j)    for j = ILO,...,IHI
 *                   = P(j)    for j = IHI+1,...,N.
 *          The order in which the interchanges are made is N to IHI+1,
 *          then 1 to ILO-1.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  The permutations consist of row and column interchanges which put
 *  the matrix in the form
 *
 *             ( T1   X   Y  )
 *     P A P = (  0   B   Z  )
 *             (  0   0   T2 )
 *
 *  where T1 and T2 are upper triangular matrices whose eigenvalues lie
 *  along the diagonal.  The column indices ILO and IHI mark the starting
 *  and ending columns of the submatrix B. Balancing consists of applying
 *  a diagonal similarity transformation inv(D) * B * D to make the
 *  1-norms of each row of B and its corresponding column nearly equal.
 *  The output matrix is
 *
 *     ( T1     X*D          Y    )
 *     (  0  inv(D)*B*D  inv(D)*Z ).
 *     (  0      0           T2   )
 *
 *  Information about the permutations P and the diagonal matrix D is
 *  returned in the vector SCALE.
 *
 *  This subroutine is based on the EISPACK routine BALANC.
 *
 *  Modified by Tzu-Yi Chen, Computer Science Division, University of
 *    California at Berkeley, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEBAL(char job, int n, double* a, int lda, int* ilo, int* ihi, double* scale)
{
    int info;
    ::F_DGEBAL(&job, &n, a, &lda, ilo, ihi, scale, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEBRD reduces a general real M-by-N matrix A to upper or lower
 *  bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
 *
 *  If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows in the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns in the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N general matrix to be reduced.
 *          On exit,
 *          if m >= n, the diagonal and the first superdiagonal are
 *            overwritten with the upper bidiagonal matrix B; the
 *            elements below the diagonal, with the array TAUQ, represent
 *            the orthogonal matrix Q as a product of elementary
 *            reflectors, and the elements above the first superdiagonal,
 *            with the array TAUP, represent the orthogonal matrix P as
 *            a product of elementary reflectors;
 *          if m < n, the diagonal and the first subdiagonal are
 *            overwritten with the lower bidiagonal matrix B; the
 *            elements below the first subdiagonal, with the array TAUQ,
 *            represent the orthogonal matrix Q as a product of
 *            elementary reflectors, and the elements above the diagonal,
 *            with the array TAUP, represent the orthogonal matrix P as
 *            a product of elementary reflectors.
 *          See Further Details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  D       (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The diagonal elements of the bidiagonal matrix B:
 *          D(i) = A(i,i).
 *
 *  E       (output) DOUBLE PRECISION array, dimension (min(M,N)-1)
 *          The off-diagonal elements of the bidiagonal matrix B:
 *          if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
 *          if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.
 *
 *  TAUQ    (output) DOUBLE PRECISION array dimension (min(M,N))
 *          The scalar factors of the elementary reflectors which
 *          represent the orthogonal matrix Q. See Further Details.
 *
 *  TAUP    (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors which
 *          represent the orthogonal matrix P. See Further Details.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of the array WORK.  LWORK >= max(1,M,N).
 *          For optimum performance LWORK >= (M+N)*NB, where NB
 *          is the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  The matrices Q and P are represented as products of elementary
 *  reflectors:
 *
 *  If m >= n,
 *
 *     Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
 *
 *  Each H(i) and G(i) has the form:
 *
 *     H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'
 *
 *  where tauq and taup are real scalars, and v and u are real vectors;
 *  v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
 *  u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
 *  tauq is stored in TAUQ(i) and taup in TAUP(i).
 *
 *  If m < n,
 *
 *     Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
 *
 *  Each H(i) and G(i) has the form:
 *
 *     H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'
 *
 *  where tauq and taup are real scalars, and v and u are real vectors;
 *  v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);
 *  u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);
 *  tauq is stored in TAUQ(i) and taup in TAUP(i).
 *
 *  The contents of A on exit are illustrated by the following examples:
 *
 *  m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
 *
 *    (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
 *    (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
 *    (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
 *    (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
 *    (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
 *    (  v1  v2  v3  v4  v5 )
 *
 *  where d and e denote diagonal and off-diagonal elements of B, vi
 *  denotes an element of the vector defining H(i), and ui an element of
 *  the vector defining G(i).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEBRD(int m, int n, double* a, int lda, double* d, double* e, double* tauq, double* taup, double* work, int lwork)
{
    int info;
    ::F_DGEBRD(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGECON estimates the reciprocal of the condition number of a general
 *  real matrix A, in either the 1-norm or the infinity-norm, using
 *  the LU factorization computed by DGETRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as
 *     RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 *  Arguments
 *  =========
 *
 *  NORM    (input) CHARACTER*1
 *          Specifies whether the 1-norm condition number or the
 *          infinity-norm condition number is required:
 *          = '1' or 'O':  1-norm;
 *          = 'I':         Infinity-norm.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The factors L and U from the factorization A = P*L*U
 *          as computed by DGETRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          If NORM = '1' or 'O', the 1-norm of the original matrix A.
 *          If NORM = 'I', the infinity-norm of the original matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(norm(A) * norm(inv(A))).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (4*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGECON(char norm, int n, double* a, int lda, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DGECON(&norm, &n, a, &lda, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEEQU computes row and column scalings intended to equilibrate an
 *  M-by-N matrix A and reduce its condition number.  R returns the row
 *  scale factors and C the column scale factors, chosen to try to make
 *  the largest element in each row and column of the matrix B with
 *  elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.
 *
 *  R(i) and C(j) are restricted to be between SMLNUM = smallest safe
 *  number and BIGNUM = largest safe number.  Use of these scaling
 *  factors is not guaranteed to reduce the condition number of A but
 *  works well in practice.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The M-by-N matrix whose equilibration factors are
 *          to be computed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  R       (output) DOUBLE PRECISION array, dimension (M)
 *          If INFO = 0 or INFO > M, R contains the row scale factors
 *          for A.
 *
 *  C       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0,  C contains the column scale factors for A.
 *
 *  ROWCND  (output) DOUBLE PRECISION
 *          If INFO = 0 or INFO > M, ROWCND contains the ratio of the
 *          smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and
 *          AMAX is neither too large nor too small, it is not worth
 *          scaling by R.
 *
 *  COLCND  (output) DOUBLE PRECISION
 *          If INFO = 0, COLCND contains the ratio of the smallest
 *          C(i) to the largest C(i).  If COLCND >= 0.1, it is not
 *          worth scaling by C.
 *
 *  AMAX    (output) DOUBLE PRECISION
 *          Absolute value of largest matrix element.  If AMAX is very
 *          close to overflow or very close to underflow, the matrix
 *          should be scaled.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i,  and i is
 *                <= M:  the i-th row of A is exactly zero
 *                >  M:  the (i-M)-th column of A is exactly zero
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEEQU(int m, int n, double* a, int lda, double* r, double* c, double* rowcnd, double* colcnd, double* amax)
{
    int info;
    ::F_DGEEQU(&m, &n, a, &lda, r, c, rowcnd, colcnd, amax, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEES computes for an N-by-N real nonsymmetric matrix A, the
 *  eigenvalues, the real Schur form T, and, optionally, the matrix of
 *  Schur vectors Z.  This gives the Schur factorization A = Z*T*(Z**T).
 *
 *  Optionally, it also orders the eigenvalues on the diagonal of the
 *  real Schur form so that selected eigenvalues are at the top left.
 *  The leading columns of Z then form an orthonormal basis for the
 *  invariant subspace corresponding to the selected eigenvalues.
 *
 *  A matrix is in real Schur form if it is upper quasi-triangular with
 *  1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in the
 *  form
 *          [  a  b  ]
 *          [  c  a  ]
 *
 *  where b*c < 0. The eigenvalues of such a block are a +- sqrt(bc).
 *
 *  Arguments
 *  =========
 *
 *  JOBVS   (input) CHARACTER*1
 *          = 'N': Schur vectors are not computed;
 *          = 'V': Schur vectors are computed.
 *
 *  SORT    (input) CHARACTER*1
 *          Specifies whether or not to order the eigenvalues on the
 *          diagonal of the Schur form.
 *          = 'N': Eigenvalues are not ordered;
 *          = 'S': Eigenvalues are ordered (see SELECT).
 *
 *  SELECT  (external procedure) LOGICAL FUNCTION of two DOUBLE PRECISION arguments
 *          SELECT must be declared EXTERNAL in the calling subroutine.
 *          If SORT = 'S', SELECT is used to select eigenvalues to sort
 *          to the top left of the Schur form.
 *          If SORT = 'N', SELECT is not referenced.
 *          An eigenvalue WR(j)+sqrt(-1)*WI(j) is selected if
 *          SELECT(WR(j),WI(j)) is true; i.e., if either one of a complex
 *          conjugate pair of eigenvalues is selected, then both complex
 *          eigenvalues are selected.
 *          Note that a selected complex eigenvalue may no longer
 *          satisfy SELECT(WR(j),WI(j)) = .TRUE. after ordering, since
 *          ordering may change the value of complex eigenvalues
 *          (especially if the eigenvalue is ill-conditioned); in this
 *          case INFO is set to N+2 (see INFO below).
 *
 *  N       (input) INTEGER
 *          The order of the matrix A. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the N-by-N matrix A.
 *          On exit, A has been overwritten by its real Schur form T.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  SDIM    (output) INTEGER
 *          If SORT = 'N', SDIM = 0.
 *          If SORT = 'S', SDIM = number of eigenvalues (after sorting)
 *                         for which SELECT is true. (Complex conjugate
 *                         pairs for which SELECT is true for either
 *                         eigenvalue count as 2.)
 *
 *  WR      (output) DOUBLE PRECISION array, dimension (N)
 *  WI      (output) DOUBLE PRECISION array, dimension (N)
 *          WR and WI contain the real and imaginary parts,
 *          respectively, of the computed eigenvalues in the same order
 *          that they appear on the diagonal of the output Schur form T.
 *          Complex conjugate pairs of eigenvalues will appear
 *          consecutively with the eigenvalue having the positive
 *          imaginary part first.
 *
 *  VS      (output) DOUBLE PRECISION array, dimension (LDVS,N)
 *          If JOBVS = 'V', VS contains the orthogonal matrix Z of Schur
 *          vectors.
 *          If JOBVS = 'N', VS is not referenced.
 *
 *  LDVS    (input) INTEGER
 *          The leading dimension of the array VS.  LDVS >= 1; if
 *          JOBVS = 'V', LDVS >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) contains the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,3*N).
 *          For good performance, LWORK must generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  BWORK   (workspace) LOGICAL array, dimension (N)
 *          Not referenced if SORT = 'N'.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value.
 *          > 0: if INFO = i, and i is
 *             <= N: the QR algorithm failed to compute all the
 *                   eigenvalues; elements 1:ILO-1 and i+1:N of WR and WI
 *                   contain those eigenvalues which have converged; if
 *                   JOBVS = 'V', VS contains the matrix which reduces A
 *                   to its partially converged Schur form.
 *             = N+1: the eigenvalues could not be reordered because some
 *                   eigenvalues were too close to separate (the problem
 *                   is very ill-conditioned);
 *             = N+2: after reordering, roundoff changed values of some
 *                   complex eigenvalues so that leading eigenvalues in
 *                   the Schur form no longer satisfy SELECT=.TRUE.  This
 *                   could also be caused by underflow due to scaling.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEES(char jobvs, char sort, int n, double* a, int lda, int* sdim, double* wr, double* wi, double* vs, int ldvs, double* work, int lwork)
{
    int info;
    ::F_DGEES(&jobvs, &sort, &n, a, &lda, sdim, wr, wi, vs, &ldvs, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEESX computes for an N-by-N real nonsymmetric matrix A, the
 *  eigenvalues, the real Schur form T, and, optionally, the matrix of
 *  Schur vectors Z.  This gives the Schur factorization A = Z*T*(Z**T).
 *
 *  Optionally, it also orders the eigenvalues on the diagonal of the
 *  real Schur form so that selected eigenvalues are at the top left;
 *  computes a reciprocal condition number for the average of the
 *  selected eigenvalues (RCONDE); and computes a reciprocal condition
 *  number for the right invariant subspace corresponding to the
 *  selected eigenvalues (RCONDV).  The leading columns of Z form an
 *  orthonormal basis for this invariant subspace.
 *
 *  For further explanation of the reciprocal condition numbers RCONDE
 *  and RCONDV, see Section 4.10 of the LAPACK Users' Guide (where
 *  these quantities are called s and sep respectively).
 *
 *  A real matrix is in real Schur form if it is upper quasi-triangular
 *  with 1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in
 *  the form
 *            [  a  b  ]
 *            [  c  a  ]
 *
 *  where b*c < 0. The eigenvalues of such a block are a +- sqrt(bc).
 *
 *  Arguments
 *  =========
 *
 *  JOBVS   (input) CHARACTER*1
 *          = 'N': Schur vectors are not computed;
 *          = 'V': Schur vectors are computed.
 *
 *  SORT    (input) CHARACTER*1
 *          Specifies whether or not to order the eigenvalues on the
 *          diagonal of the Schur form.
 *          = 'N': Eigenvalues are not ordered;
 *          = 'S': Eigenvalues are ordered (see SELECT).
 *
 *  SELECT  (external procedure) LOGICAL FUNCTION of two DOUBLE PRECISION arguments
 *          SELECT must be declared EXTERNAL in the calling subroutine.
 *          If SORT = 'S', SELECT is used to select eigenvalues to sort
 *          to the top left of the Schur form.
 *          If SORT = 'N', SELECT is not referenced.
 *          An eigenvalue WR(j)+sqrt(-1)*WI(j) is selected if
 *          SELECT(WR(j),WI(j)) is true; i.e., if either one of a
 *          complex conjugate pair of eigenvalues is selected, then both
 *          are.  Note that a selected complex eigenvalue may no longer
 *          satisfy SELECT(WR(j),WI(j)) = .TRUE. after ordering, since
 *          ordering may change the value of complex eigenvalues
 *          (especially if the eigenvalue is ill-conditioned); in this
 *          case INFO may be set to N+3 (see INFO below).
 *
 *  SENSE   (input) CHARACTER*1
 *          Determines which reciprocal condition numbers are computed.
 *          = 'N': None are computed;
 *          = 'E': Computed for average of selected eigenvalues only;
 *          = 'V': Computed for selected right invariant subspace only;
 *          = 'B': Computed for both.
 *          If SENSE = 'E', 'V' or 'B', SORT must equal 'S'.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the N-by-N matrix A.
 *          On exit, A is overwritten by its real Schur form T.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  SDIM    (output) INTEGER
 *          If SORT = 'N', SDIM = 0.
 *          If SORT = 'S', SDIM = number of eigenvalues (after sorting)
 *                         for which SELECT is true. (Complex conjugate
 *                         pairs for which SELECT is true for either
 *                         eigenvalue count as 2.)
 *
 *  WR      (output) DOUBLE PRECISION array, dimension (N)
 *  WI      (output) DOUBLE PRECISION array, dimension (N)
 *          WR and WI contain the real and imaginary parts, respectively,
 *          of the computed eigenvalues, in the same order that they
 *          appear on the diagonal of the output Schur form T.  Complex
 *          conjugate pairs of eigenvalues appear consecutively with the
 *          eigenvalue having the positive imaginary part first.
 *
 *  VS      (output) DOUBLE PRECISION array, dimension (LDVS,N)
 *          If JOBVS = 'V', VS contains the orthogonal matrix Z of Schur
 *          vectors.
 *          If JOBVS = 'N', VS is not referenced.
 *
 *  LDVS    (input) INTEGER
 *          The leading dimension of the array VS.  LDVS >= 1, and if
 *          JOBVS = 'V', LDVS >= N.
 *
 *  RCONDE  (output) DOUBLE PRECISION
 *          If SENSE = 'E' or 'B', RCONDE contains the reciprocal
 *          condition number for the average of the selected eigenvalues.
 *          Not referenced if SENSE = 'N' or 'V'.
 *
 *  RCONDV  (output) DOUBLE PRECISION
 *          If SENSE = 'V' or 'B', RCONDV contains the reciprocal
 *          condition number for the selected right invariant subspace.
 *          Not referenced if SENSE = 'N' or 'E'.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,3*N).
 *          Also, if SENSE = 'E' or 'V' or 'B',
 *          LWORK >= N+2*SDIM*(N-SDIM), where SDIM is the number of
 *          selected eigenvalues computed by this routine.  Note that
 *          N+2*SDIM*(N-SDIM) <= N+N*N/2. Note also that an error is only
 *          returned if LWORK < max(1,3*N), but if SENSE = 'E' or 'V' or
 *          'B' this may not be large enough.
 *          For good performance, LWORK must generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates upper bounds on the optimal sizes of the
 *          arrays WORK and IWORK, returns these values as the first
 *          entries of the WORK and IWORK arrays, and no error messages
 *          related to LWORK or LIWORK are issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          LIWORK >= 1; if SENSE = 'V' or 'B', LIWORK >= SDIM*(N-SDIM).
 *          Note that SDIM*(N-SDIM) <= N*N/4. Note also that an error is
 *          only returned if LIWORK < 1, but if SENSE = 'V' or 'B' this
 *          may not be large enough.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates upper bounds on the optimal sizes of
 *          the arrays WORK and IWORK, returns these values as the first
 *          entries of the WORK and IWORK arrays, and no error messages
 *          related to LWORK or LIWORK are issued by XERBLA.
 *
 *  BWORK   (workspace) LOGICAL array, dimension (N)
 *          Not referenced if SORT = 'N'.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value.
 *          > 0: if INFO = i, and i is
 *             <= N: the QR algorithm failed to compute all the
 *                   eigenvalues; elements 1:ILO-1 and i+1:N of WR and WI
 *                   contain those eigenvalues which have converged; if
 *                   JOBVS = 'V', VS contains the transformation which
 *                   reduces A to its partially converged Schur form.
 *             = N+1: the eigenvalues could not be reordered because some
 *                   eigenvalues were too close to separate (the problem
 *                   is very ill-conditioned);
 *             = N+2: after reordering, roundoff changed values of some
 *                   complex eigenvalues so that leading eigenvalues in
 *                   the Schur form no longer satisfy SELECT=.TRUE.  This
 *                   could also be caused by underflow due to scaling.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEESX(char jobvs, char sort, char sense, int n, double* a, int lda, int* sdim, double* wr, double* wi, double* vs, int ldvs, double* rconde, double* rcondv, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DGEESX(&jobvs, &sort, &sense, &n, a, &lda, sdim, wr, wi, vs, &ldvs, rconde, rcondv, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEEV computes for an N-by-N real nonsymmetric matrix A, the
 *  eigenvalues and, optionally, the left and/or right eigenvectors.
 *
 *  The right eigenvector v(j) of A satisfies
 *                   A * v(j) = lambda(j) * v(j)
 *  where lambda(j) is its eigenvalue.
 *  The left eigenvector u(j) of A satisfies
 *                u(j)**H * A = lambda(j) * u(j)**H
 *  where u(j)**H denotes the conjugate transpose of u(j).
 *
 *  The computed eigenvectors are normalized to have Euclidean norm
 *  equal to 1 and largest component real.
 *
 *  Arguments
 *  =========
 *
 *  JOBVL   (input) CHARACTER*1
 *          = 'N': left eigenvectors of A are not computed;
 *          = 'V': left eigenvectors of A are computed.
 *
 *  JOBVR   (input) CHARACTER*1
 *          = 'N': right eigenvectors of A are not computed;
 *          = 'V': right eigenvectors of A are computed.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the N-by-N matrix A.
 *          On exit, A has been overwritten.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  WR      (output) DOUBLE PRECISION array, dimension (N)
 *  WI      (output) DOUBLE PRECISION array, dimension (N)
 *          WR and WI contain the real and imaginary parts,
 *          respectively, of the computed eigenvalues.  Complex
 *          conjugate pairs of eigenvalues appear consecutively
 *          with the eigenvalue having the positive imaginary part
 *          first.
 *
 *  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N)
 *          If JOBVL = 'V', the left eigenvectors u(j) are stored one
 *          after another in the columns of VL, in the same order
 *          as their eigenvalues.
 *          If JOBVL = 'N', VL is not referenced.
 *          If the j-th eigenvalue is real, then u(j) = VL(:,j),
 *          the j-th column of VL.
 *          If the j-th and (j+1)-st eigenvalues form a complex
 *          conjugate pair, then u(j) = VL(:,j) + i*VL(:,j+1) and
 *          u(j+1) = VL(:,j) - i*VL(:,j+1).
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the array VL.  LDVL >= 1; if
 *          JOBVL = 'V', LDVL >= N.
 *
 *  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N)
 *          If JOBVR = 'V', the right eigenvectors v(j) are stored one
 *          after another in the columns of VR, in the same order
 *          as their eigenvalues.
 *          If JOBVR = 'N', VR is not referenced.
 *          If the j-th eigenvalue is real, then v(j) = VR(:,j),
 *          the j-th column of VR.
 *          If the j-th and (j+1)-st eigenvalues form a complex
 *          conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and
 *          v(j+1) = VR(:,j) - i*VR(:,j+1).
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR.  LDVR >= 1; if
 *          JOBVR = 'V', LDVR >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,3*N), and
 *          if JOBVL = 'V' or JOBVR = 'V', LWORK >= 4*N.  For good
 *          performance, LWORK must generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, the QR algorithm failed to compute all the
 *                eigenvalues, and no eigenvectors have been computed;
 *                elements i+1:N of WR and WI contain eigenvalues which
 *                have converged.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEEV(char jobvl, char jobvr, int n, double* a, int lda, double* wr, double* wi, double* vl, int ldvl, double* vr, int ldvr, double* work, int lwork)
{
    int info;
    ::F_DGEEV(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEEVX computes for an N-by-N real nonsymmetric matrix A, the
 *  eigenvalues and, optionally, the left and/or right eigenvectors.
 *
 *  Optionally also, it computes a balancing transformation to improve
 *  the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
 *  SCALE, and ABNRM), reciprocal condition numbers for the eigenvalues
 *  (RCONDE), and reciprocal condition numbers for the right
 *  eigenvectors (RCONDV).
 *
 *  The right eigenvector v(j) of A satisfies
 *                   A * v(j) = lambda(j) * v(j)
 *  where lambda(j) is its eigenvalue.
 *  The left eigenvector u(j) of A satisfies
 *                u(j)**H * A = lambda(j) * u(j)**H
 *  where u(j)**H denotes the conjugate transpose of u(j).
 *
 *  The computed eigenvectors are normalized to have Euclidean norm
 *  equal to 1 and largest component real.
 *
 *  Balancing a matrix means permuting the rows and columns to make it
 *  more nearly upper triangular, and applying a diagonal similarity
 *  transformation D * A * D**(-1), where D is a diagonal matrix, to
 *  make its rows and columns closer in norm and the condition numbers
 *  of its eigenvalues and eigenvectors smaller.  The computed
 *  reciprocal condition numbers correspond to the balanced matrix.
 *  Permuting rows and columns will not change the condition numbers
 *  (in exact arithmetic) but diagonal scaling will.  For further
 *  explanation of balancing, see section 4.10.2 of the LAPACK
 *  Users' Guide.
 *
 *  Arguments
 *  =========
 *
 *  BALANC  (input) CHARACTER*1
 *          Indicates how the input matrix should be diagonally scaled
 *          and/or permuted to improve the conditioning of its
 *          eigenvalues.
 *          = 'N': Do not diagonally scale or permute;
 *          = 'P': Perform permutations to make the matrix more nearly
 *                 upper triangular. Do not diagonally scale;
 *          = 'S': Diagonally scale the matrix, i.e. replace A by
 *                 D*A*D**(-1), where D is a diagonal matrix chosen
 *                 to make the rows and columns of A more equal in
 *                 norm. Do not permute;
 *          = 'B': Both diagonally scale and permute A.
 *
 *          Computed reciprocal condition numbers will be for the matrix
 *          after balancing and/or permuting. Permuting does not change
 *          condition numbers (in exact arithmetic), but balancing does.
 *
 *  JOBVL   (input) CHARACTER*1
 *          = 'N': left eigenvectors of A are not computed;
 *          = 'V': left eigenvectors of A are computed.
 *          If SENSE = 'E' or 'B', JOBVL must = 'V'.
 *
 *  JOBVR   (input) CHARACTER*1
 *          = 'N': right eigenvectors of A are not computed;
 *          = 'V': right eigenvectors of A are computed.
 *          If SENSE = 'E' or 'B', JOBVR must = 'V'.
 *
 *  SENSE   (input) CHARACTER*1
 *          Determines which reciprocal condition numbers are computed.
 *          = 'N': None are computed;
 *          = 'E': Computed for eigenvalues only;
 *          = 'V': Computed for right eigenvectors only;
 *          = 'B': Computed for eigenvalues and right eigenvectors.
 *
 *          If SENSE = 'E' or 'B', both left and right eigenvectors
 *          must also be computed (JOBVL = 'V' and JOBVR = 'V').
 *
 *  N       (input) INTEGER
 *          The order of the matrix A. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the N-by-N matrix A.
 *          On exit, A has been overwritten.  If JOBVL = 'V' or
 *          JOBVR = 'V', A contains the real Schur form of the balanced
 *          version of the input matrix A.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  WR      (output) DOUBLE PRECISION array, dimension (N)
 *  WI      (output) DOUBLE PRECISION array, dimension (N)
 *          WR and WI contain the real and imaginary parts,
 *          respectively, of the computed eigenvalues.  Complex
 *          conjugate pairs of eigenvalues will appear consecutively
 *          with the eigenvalue having the positive imaginary part
 *          first.
 *
 *  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N)
 *          If JOBVL = 'V', the left eigenvectors u(j) are stored one
 *          after another in the columns of VL, in the same order
 *          as their eigenvalues.
 *          If JOBVL = 'N', VL is not referenced.
 *          If the j-th eigenvalue is real, then u(j) = VL(:,j),
 *          the j-th column of VL.
 *          If the j-th and (j+1)-st eigenvalues form a complex
 *          conjugate pair, then u(j) = VL(:,j) + i*VL(:,j+1) and
 *          u(j+1) = VL(:,j) - i*VL(:,j+1).
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the array VL.  LDVL >= 1; if
 *          JOBVL = 'V', LDVL >= N.
 *
 *  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N)
 *          If JOBVR = 'V', the right eigenvectors v(j) are stored one
 *          after another in the columns of VR, in the same order
 *          as their eigenvalues.
 *          If JOBVR = 'N', VR is not referenced.
 *          If the j-th eigenvalue is real, then v(j) = VR(:,j),
 *          the j-th column of VR.
 *          If the j-th and (j+1)-st eigenvalues form a complex
 *          conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and
 *          v(j+1) = VR(:,j) - i*VR(:,j+1).
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR.  LDVR >= 1, and if
 *          JOBVR = 'V', LDVR >= N.
 *
 *  ILO     (output) INTEGER
 *  IHI     (output) INTEGER
 *          ILO and IHI are integer values determined when A was
 *          balanced.  The balanced A(i,j) = 0 if I > J and
 *          J = 1,...,ILO-1 or I = IHI+1,...,N.
 *
 *  SCALE   (output) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and scaling factors applied
 *          when balancing A.  If P(j) is the index of the row and column
 *          interchanged with row and column j, and D(j) is the scaling
 *          factor applied to row and column j, then
 *          SCALE(J) = P(J),    for J = 1,...,ILO-1
 *                   = D(J),    for J = ILO,...,IHI
 *                   = P(J)     for J = IHI+1,...,N.
 *          The order in which the interchanges are made is N to IHI+1,
 *          then 1 to ILO-1.
 *
 *  ABNRM   (output) DOUBLE PRECISION
 *          The one-norm of the balanced matrix (the maximum
 *          of the sum of absolute values of elements of any column).
 *
 *  RCONDE  (output) DOUBLE PRECISION array, dimension (N)
 *          RCONDE(j) is the reciprocal condition number of the j-th
 *          eigenvalue.
 *
 *  RCONDV  (output) DOUBLE PRECISION array, dimension (N)
 *          RCONDV(j) is the reciprocal condition number of the j-th
 *          right eigenvector.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.   If SENSE = 'N' or 'E',
 *          LWORK >= max(1,2*N), and if JOBVL = 'V' or JOBVR = 'V',
 *          LWORK >= 3*N.  If SENSE = 'V' or 'B', LWORK >= N*(N+6).
 *          For good performance, LWORK must generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (2*N-2)
 *          If SENSE = 'N' or 'E', not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, the QR algorithm failed to compute all the
 *                eigenvalues, and no eigenvectors or condition numbers
 *                have been computed; elements 1:ILO-1 and i+1:N of WR
 *                and WI contain eigenvalues which have converged.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEEVX(char balanc, char jobvl, char jobvr, char sense, int n, double* a, int lda, double* wr, double* wi, double* vl, int ldvl, double* vr, int ldvr, int* ilo, int* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, double* work, int lwork, int* iwork)
{
    int info;
    ::F_DGEEVX(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, &lwork, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  This routine is deprecated and has been replaced by routine DGGES.
 *
 *  DGEGS computes the eigenvalues, real Schur form, and, optionally,
 *  left and or/right Schur vectors of a real matrix pair (A,B).
 *  Given two square matrices A and B, the generalized real Schur
 *  factorization has the form
 *
 *    A = Q*S*Z**T,  B = Q*T*Z**T
 *
 *  where Q and Z are orthogonal matrices, T is upper triangular, and S
 *  is an upper quasi-triangular matrix with 1-by-1 and 2-by-2 diagonal
 *  blocks, the 2-by-2 blocks corresponding to complex conjugate pairs
 *  of eigenvalues of (A,B).  The columns of Q are the left Schur vectors
 *  and the columns of Z are the right Schur vectors.
 *
 *  If only the eigenvalues of (A,B) are needed, the driver routine
 *  DGEGV should be used instead.  See DGEGV for a description of the
 *  eigenvalues of the generalized nonsymmetric eigenvalue problem
 *  (GNEP).
 *
 *  Arguments
 *  =========
 *
 *  JOBVSL  (input) CHARACTER*1
 *          = 'N':  do not compute the left Schur vectors;
 *          = 'V':  compute the left Schur vectors (returned in VSL).
 *
 *  JOBVSR  (input) CHARACTER*1
 *          = 'N':  do not compute the right Schur vectors;
 *          = 'V':  compute the right Schur vectors (returned in VSR).
 *
 *  N       (input) INTEGER
 *          The order of the matrices A, B, VSL, and VSR.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the matrix A.
 *          On exit, the upper quasi-triangular matrix S from the
 *          generalized real Schur factorization.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the matrix B.
 *          On exit, the upper triangular matrix T from the generalized
 *          real Schur factorization.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of B.  LDB >= max(1,N).
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *          The real parts of each scalar alpha defining an eigenvalue
 *          of GNEP.
 *
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *          The imaginary parts of each scalar alpha defining an
 *          eigenvalue of GNEP.  If ALPHAI(j) is zero, then the j-th
 *          eigenvalue is real; if positive, then the j-th and (j+1)-st
 *          eigenvalues are a complex conjugate pair, with
 *          ALPHAI(j+1) = -ALPHAI(j).
 *
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          The scalars beta that define the eigenvalues of GNEP.
 *          Together, the quantities alpha = (ALPHAR(j),ALPHAI(j)) and
 *          beta = BETA(j) represent the j-th eigenvalue of the matrix
 *          pair (A,B), in one of the forms lambda = alpha/beta or
 *          mu = beta/alpha.  Since either lambda or mu may overflow,
 *          they should not, in general, be computed.
 *
 *  VSL     (output) DOUBLE PRECISION array, dimension (LDVSL,N)
 *          If JOBVSL = 'V', the matrix of left Schur vectors Q.
 *          Not referenced if JOBVSL = 'N'.
 *
 *  LDVSL   (input) INTEGER
 *          The leading dimension of the matrix VSL. LDVSL >=1, and
 *          if JOBVSL = 'V', LDVSL >= N.
 *
 *  VSR     (output) DOUBLE PRECISION array, dimension (LDVSR,N)
 *          If JOBVSR = 'V', the matrix of right Schur vectors Z.
 *          Not referenced if JOBVSR = 'N'.
 *
 *  LDVSR   (input) INTEGER
 *          The leading dimension of the matrix VSR. LDVSR >= 1, and
 *          if JOBVSR = 'V', LDVSR >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,4*N).
 *          For good performance, LWORK must generally be larger.
 *          To compute the optimal value of LWORK, call ILAENV to get
 *          blocksizes (for DGEQRF, DORMQR, and DORGQR.)  Then compute:
 *          NB  -- MAX of the blocksizes for DGEQRF, DORMQR, and DORGQR
 *          The optimal LWORK is  2*N + N*(NB+1).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1,...,N:
 *                The QZ iteration failed.  (A,B) are not in Schur
 *                form, but ALPHAR(j), ALPHAI(j), and BETA(j) should
 *                be correct for j=INFO+1,...,N.
 *          > N:  errors that usually indicate LAPACK problems:
 *                =N+1: error return from DGGBAL
 *                =N+2: error return from DGEQRF
 *                =N+3: error return from DORMQR
 *                =N+4: error return from DORGQR
 *                =N+5: error return from DGGHRD
 *                =N+6: error return from DHGEQZ (other than failed
 *                                                iteration)
 *                =N+7: error return from DGGBAK (computing VSL)
 *                =N+8: error return from DGGBAK (computing VSR)
 *                =N+9: error return from DLASCL (various places)
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEGS(char jobvsl, char jobvsr, int n, double* a, int lda, double* b, int ldb, double* alphar, double* alphai, double* beta, double* vsl, int ldvsl, double* vsr, int ldvsr, double* work, int lwork)
{
    int info;
    ::F_DGEGS(&jobvsl, &jobvsr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vsl, &ldvsl, vsr, &ldvsr, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  This routine is deprecated and has been replaced by routine DGGEV.
 *
 *  DGEGV computes the eigenvalues and, optionally, the left and/or right
 *  eigenvectors of a real matrix pair (A,B).
 *  Given two square matrices A and B,
 *  the generalized nonsymmetric eigenvalue problem (GNEP) is to find the
 *  eigenvalues lambda and corresponding (non-zero) eigenvectors x such
 *  that
 *
 *     A*x = lambda*B*x.
 *
 *  An alternate form is to find the eigenvalues mu and corresponding
 *  eigenvectors y such that
 *
 *     mu*A*y = B*y.
 *
 *  These two forms are equivalent with mu = 1/lambda and x = y if
 *  neither lambda nor mu is zero.  In order to deal with the case that
 *  lambda or mu is zero or small, two values alpha and beta are returned
 *  for each eigenvalue, such that lambda = alpha/beta and
 *  mu = beta/alpha.
 *
 *  The vectors x and y in the above equations are right eigenvectors of
 *  the matrix pair (A,B).  Vectors u and v satisfying
 *
 *     u**H*A = lambda*u**H*B  or  mu*v**H*A = v**H*B
 *
 *  are left eigenvectors of (A,B).
 *
 *  Note: this routine performs "full balancing" on A and B -- see
 *  "Further Details", below.
 *
 *  Arguments
 *  =========
 *
 *  JOBVL   (input) CHARACTER*1
 *          = 'N':  do not compute the left generalized eigenvectors;
 *          = 'V':  compute the left generalized eigenvectors (returned
 *                  in VL).
 *
 *  JOBVR   (input) CHARACTER*1
 *          = 'N':  do not compute the right generalized eigenvectors;
 *          = 'V':  compute the right generalized eigenvectors (returned
 *                  in VR).
 *
 *  N       (input) INTEGER
 *          The order of the matrices A, B, VL, and VR.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the matrix A.
 *          If JOBVL = 'V' or JOBVR = 'V', then on exit A
 *          contains the real Schur form of A from the generalized Schur
 *          factorization of the pair (A,B) after balancing.
 *          If no eigenvectors were computed, then only the diagonal
 *          blocks from the Schur form will be correct.  See DGGHRD and
 *          DHGEQZ for details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the matrix B.
 *          If JOBVL = 'V' or JOBVR = 'V', then on exit B contains the
 *          upper triangular matrix obtained from B in the generalized
 *          Schur factorization of the pair (A,B) after balancing.
 *          If no eigenvectors were computed, then only those elements of
 *          B corresponding to the diagonal blocks from the Schur form of
 *          A will be correct.  See DGGHRD and DHGEQZ for details.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of B.  LDB >= max(1,N).
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *          The real parts of each scalar alpha defining an eigenvalue of
 *          GNEP.
 *
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *          The imaginary parts of each scalar alpha defining an
 *          eigenvalue of GNEP.  If ALPHAI(j) is zero, then the j-th
 *          eigenvalue is real; if positive, then the j-th and
 *          (j+1)-st eigenvalues are a complex conjugate pair, with
 *          ALPHAI(j+1) = -ALPHAI(j).
 *
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          The scalars beta that define the eigenvalues of GNEP.
 *
 *          Together, the quantities alpha = (ALPHAR(j),ALPHAI(j)) and
 *          beta = BETA(j) represent the j-th eigenvalue of the matrix
 *          pair (A,B), in one of the forms lambda = alpha/beta or
 *          mu = beta/alpha.  Since either lambda or mu may overflow,
 *          they should not, in general, be computed.
 *
 *  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N)
 *          If JOBVL = 'V', the left eigenvectors u(j) are stored
 *          in the columns of VL, in the same order as their eigenvalues.
 *          If the j-th eigenvalue is real, then u(j) = VL(:,j).
 *          If the j-th and (j+1)-st eigenvalues form a complex conjugate
 *          pair, then
 *             u(j) = VL(:,j) + i*VL(:,j+1)
 *          and
 *            u(j+1) = VL(:,j) - i*VL(:,j+1).
 *
 *          Each eigenvector is scaled so that its largest component has
 *          abs(real part) + abs(imag. part) = 1, except for eigenvectors
 *          corresponding to an eigenvalue with alpha = beta = 0, which
 *          are set to zero.
 *          Not referenced if JOBVL = 'N'.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the matrix VL. LDVL >= 1, and
 *          if JOBVL = 'V', LDVL >= N.
 *
 *  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N)
 *          If JOBVR = 'V', the right eigenvectors x(j) are stored
 *          in the columns of VR, in the same order as their eigenvalues.
 *          If the j-th eigenvalue is real, then x(j) = VR(:,j).
 *          If the j-th and (j+1)-st eigenvalues form a complex conjugate
 *          pair, then
 *            x(j) = VR(:,j) + i*VR(:,j+1)
 *          and
 *            x(j+1) = VR(:,j) - i*VR(:,j+1).
 *
 *          Each eigenvector is scaled so that its largest component has
 *          abs(real part) + abs(imag. part) = 1, except for eigenvalues
 *          corresponding to an eigenvalue with alpha = beta = 0, which
 *          are set to zero.
 *          Not referenced if JOBVR = 'N'.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the matrix VR. LDVR >= 1, and
 *          if JOBVR = 'V', LDVR >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,8*N).
 *          For good performance, LWORK must generally be larger.
 *          To compute the optimal value of LWORK, call ILAENV to get
 *          blocksizes (for DGEQRF, DORMQR, and DORGQR.)  Then compute:
 *          NB  -- MAX of the blocksizes for DGEQRF, DORMQR, and DORGQR;
 *          The optimal LWORK is:
 *              2*N + MAX( 6*N, N*(NB+1) ).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1,...,N:
 *                The QZ iteration failed.  No eigenvectors have been
 *                calculated, but ALPHAR(j), ALPHAI(j), and BETA(j)
 *                should be correct for j=INFO+1,...,N.
 *          > N:  errors that usually indicate LAPACK problems:
 *                =N+1: error return from DGGBAL
 *                =N+2: error return from DGEQRF
 *                =N+3: error return from DORMQR
 *                =N+4: error return from DORGQR
 *                =N+5: error return from DGGHRD
 *                =N+6: error return from DHGEQZ (other than failed
 *                                                iteration)
 *                =N+7: error return from DTGEVC
 *                =N+8: error return from DGGBAK (computing VL)
 *                =N+9: error return from DGGBAK (computing VR)
 *                =N+10: error return from DLASCL (various calls)
 *
 *  Further Details
 *  ===============
 *
 *  Balancing
 *  ---------
 *
 *  This driver calls DGGBAL to both permute and scale rows and columns
 *  of A and B.  The permutations PL and PR are chosen so that PL*A*PR
 *  and PL*B*R will be upper triangular except for the diagonal blocks
 *  A(i:j,i:j) and B(i:j,i:j), with i and j as close together as
 *  possible.  The diagonal scaling matrices DL and DR are chosen so
 *  that the pair  DL*PL*A*PR*DR, DL*PL*B*PR*DR have elements close to
 *  one (except for the elements that start out zero.)
 *
 *  After the eigenvalues and eigenvectors of the balanced matrices
 *  have been computed, DGGBAK transforms the eigenvectors back to what
 *  they would have been (in perfect arithmetic) if they had not been
 *  balanced.
 *
 *  Contents of A and B on Exit
 *  -------- -- - --- - -- ----
 *
 *  If any eigenvectors are computed (either JOBVL='V' or JOBVR='V' or
 *  both), then on exit the arrays A and B will contain the real Schur
 *  form[*] of the "balanced" versions of A and B.  If no eigenvectors
 *  are computed, then only the diagonal blocks will be correct.
 *
 *  [*] See DHGEQZ, DGEGS, or read the book "Matrix Computations",
 *      by Golub & van Loan, pub. by Johns Hopkins U. Press.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEGV(char jobvl, char jobvr, int n, double* a, int lda, double* b, int ldb, double* alphar, double* alphai, double* beta, double* vl, int ldvl, double* vr, int ldvr, double* work, int lwork)
{
    int info;
    ::F_DGEGV(&jobvl, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEHRD reduces a real general matrix A to upper Hessenberg form H by
 *  an orthogonal similarity transformation:  Q' * A * Q = H .
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          It is assumed that A is already upper triangular in rows
 *          and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
 *          set by a previous call to DGEBAL; otherwise they should be
 *          set to 1 and N respectively. See Further Details.
 *          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the N-by-N general matrix to be reduced.
 *          On exit, the upper triangle and the first subdiagonal of A
 *          are overwritten with the upper Hessenberg matrix H, and the
 *          elements below the first subdiagonal, with the array TAU,
 *          represent the orthogonal matrix Q as a product of elementary
 *          reflectors. See Further Details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (N-1)
 *          The scalar factors of the elementary reflectors (see Further
 *          Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
 *          zero.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of the array WORK.  LWORK >= max(1,N).
 *          For optimum performance LWORK >= N*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of (ihi-ilo) elementary
 *  reflectors
 *
 *     Q = H(ilo) H(ilo+1) . . . H(ihi-1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
 *  exit in A(i+2:ihi,i), and tau in TAU(i).
 *
 *  The contents of A are illustrated by the following example, with
 *  n = 7, ilo = 2 and ihi = 6:
 *
 *  on entry,                        on exit,
 *
 *  ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
 *  (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
 *  (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
 *  (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
 *  (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
 *  (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
 *  (                         a )    (                          a )
 *
 *  where a denotes an element of the original matrix A, h denotes a
 *  modified element of the upper Hessenberg matrix H, and vi denotes an
 *  element of the vector defining H(i).
 *
 *  This file is a slight modification of LAPACK-3.0's DGEHRD
 *  subroutine incorporating improvements proposed by Quintana-Orti and
 *  Van de Geijn (2006). (See DLAHR2.)
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEHRD(int n, int ilo, int ihi, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DGEHRD(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGELQF computes an LQ factorization of a real M-by-N matrix A:
 *  A = L * Q.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, the elements on and below the diagonal of the array
 *          contain the m-by-min(m,n) lower trapezoidal matrix L (L is
 *          lower triangular if m <= n); the elements above the diagonal,
 *          with the array TAU, represent the orthogonal matrix Q as a
 *          product of elementary reflectors (see Further Details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors (see Further
 *          Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,M).
 *          For optimum performance LWORK >= M*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(k) . . . H(2) H(1), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
 *  and tau in TAU(i).
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGELQF(int m, int n, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DGELQF(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGELS solves overdetermined or underdetermined real linear systems
 *  involving an M-by-N matrix A, or its transpose, using a QR or LQ
 *  factorization of A.  It is assumed that A has full rank.
 *
 *  The following options are provided:
 *
 *  1. If TRANS = 'N' and m >= n:  find the least squares solution of
 *     an overdetermined system, i.e., solve the least squares problem
 *                  minimize || B - A*X ||.
 *
 *  2. If TRANS = 'N' and m < n:  find the minimum norm solution of
 *     an underdetermined system A * X = B.
 *
 *  3. If TRANS = 'T' and m >= n:  find the minimum norm solution of
 *     an undetermined system A**T * X = B.
 *
 *  4. If TRANS = 'T' and m < n:  find the least squares solution of
 *     an overdetermined system, i.e., solve the least squares problem
 *                  minimize || B - A**T * X ||.
 *
 *  Several right hand side vectors b and solution vectors x can be
 *  handled in a single call; they are stored as the columns of the
 *  M-by-NRHS right hand side matrix B and the N-by-NRHS solution
 *  matrix X.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N': the linear system involves A;
 *          = 'T': the linear system involves A**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of
 *          columns of the matrices B and X. NRHS >=0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit,
 *            if M >= N, A is overwritten by details of its QR
 *                       factorization as returned by DGEQRF;
 *            if M <  N, A is overwritten by details of its LQ
 *                       factorization as returned by DGELQF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the matrix B of right hand side vectors, stored
 *          columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS
 *          if TRANS = 'T'.
 *          On exit, if INFO = 0, B is overwritten by the solution
 *          vectors, stored columnwise:
 *          if TRANS = 'N' and m >= n, rows 1 to n of B contain the least
 *          squares solution vectors; the residual sum of squares for the
 *          solution in each column is given by the sum of squares of
 *          elements N+1 to M in that column;
 *          if TRANS = 'N' and m < n, rows 1 to N of B contain the
 *          minimum norm solution vectors;
 *          if TRANS = 'T' and m >= n, rows 1 to M of B contain the
 *          minimum norm solution vectors;
 *          if TRANS = 'T' and m < n, rows 1 to M of B contain the
 *          least squares solution vectors; the residual sum of squares
 *          for the solution in each column is given by the sum of
 *          squares of elements M+1 to N in that column.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= MAX(1,M,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          LWORK >= max( 1, MN + max( MN, NRHS ) ).
 *          For optimal performance,
 *          LWORK >= max( 1, MN + max( MN, NRHS )*NB ).
 *          where MN = min(M,N) and NB is the optimum block size.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO =  i, the i-th diagonal element of the
 *                triangular factor of A is zero, so that A does not have
 *                full rank; the least squares solution could not be
 *                computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGELS(char trans, int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* work, int lwork)
{
    int info;
    ::F_DGELS(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGELSD computes the minimum-norm solution to a real linear least
 *  squares problem:
 *      minimize 2-norm(| b - A*x |)
 *  using the singular value decomposition (SVD) of A. A is an M-by-N
 *  matrix which may be rank-deficient.
 *
 *  Several right hand side vectors b and solution vectors x can be
 *  handled in a single call; they are stored as the columns of the
 *  M-by-NRHS right hand side matrix B and the N-by-NRHS solution
 *  matrix X.
 *
 *  The problem is solved in three steps:
 *  (1) Reduce the coefficient matrix A to bidiagonal form with
 *      Householder transformations, reducing the original problem
 *      into a "bidiagonal least squares problem" (BLS)
 *  (2) Solve the BLS using a divide and conquer approach.
 *  (3) Apply back all the Householder tranformations to solve
 *      the original least squares problem.
 *
 *  The effective rank of A is determined by treating as zero those
 *  singular values which are less than RCOND times the largest singular
 *  value.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of A. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of A. N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X. NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, A has been destroyed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the M-by-NRHS right hand side matrix B.
 *          On exit, B is overwritten by the N-by-NRHS solution
 *          matrix X.  If m >= n and RANK = n, the residual
 *          sum-of-squares for the solution in the i-th column is given
 *          by the sum of squares of elements n+1:m in that column.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,max(M,N)).
 *
 *  S       (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The singular values of A in decreasing order.
 *          The condition number of A in the 2-norm = S(1)/S(min(m,n)).
 *
 *  RCOND   (input) DOUBLE PRECISION
 *          RCOND is used to determine the effective rank of A.
 *          Singular values S(i) <= RCOND*S(1) are treated as zero.
 *          If RCOND < 0, machine precision is used instead.
 *
 *  RANK    (output) INTEGER
 *          The effective rank of A, i.e., the number of singular values
 *          which are greater than RCOND*S(1).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK must be at least 1.
 *          The exact minimum amount of workspace needed depends on M,
 *          N and NRHS. As long as LWORK is at least
 *              12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2,
 *          if M is greater than or equal to N or
 *              12*M + 2*M*SMLSIZ + 8*M*NLVL + M*NRHS + (SMLSIZ+1)**2,
 *          if M is less than N, the code will execute correctly.
 *          SMLSIZ is returned by ILAENV and is equal to the maximum
 *          size of the subproblems at the bottom of the computation
 *          tree (usually about 25), and
 *             NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 )
 *          For good performance, LWORK should generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
 *          LIWORK >= max(1, 3 * MINMN * NLVL + 11 * MINMN),
 *          where MINMN = MIN( M,N ).
 *          On exit, if INFO = 0, IWORK(1) returns the minimum LIWORK.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  the algorithm for computing the SVD failed to converge;
 *                if INFO = i, i off-diagonal elements of an intermediate
 *                bidiagonal form did not converge to zero.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Ming Gu and Ren-Cang Li, Computer Science Division, University of
 *       California at Berkeley, USA
 *     Osni Marques, LBNL/NERSC, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGELSD(int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank, double* work, int lwork, int* iwork)
{
    int info;
    ::F_DGELSD(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGELSS computes the minimum norm solution to a real linear least
 *  squares problem:
 *
 *  Minimize 2-norm(| b - A*x |).
 *
 *  using the singular value decomposition (SVD) of A. A is an M-by-N
 *  matrix which may be rank-deficient.
 *
 *  Several right hand side vectors b and solution vectors x can be
 *  handled in a single call; they are stored as the columns of the
 *  M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix
 *  X.
 *
 *  The effective rank of A is determined by treating as zero those
 *  singular values which are less than RCOND times the largest singular
 *  value.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A. N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X. NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, the first min(m,n) rows of A are overwritten with
 *          its right singular vectors, stored rowwise.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the M-by-NRHS right hand side matrix B.
 *          On exit, B is overwritten by the N-by-NRHS solution
 *          matrix X.  If m >= n and RANK = n, the residual
 *          sum-of-squares for the solution in the i-th column is given
 *          by the sum of squares of elements n+1:m in that column.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,max(M,N)).
 *
 *  S       (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The singular values of A in decreasing order.
 *          The condition number of A in the 2-norm = S(1)/S(min(m,n)).
 *
 *  RCOND   (input) DOUBLE PRECISION
 *          RCOND is used to determine the effective rank of A.
 *          Singular values S(i) <= RCOND*S(1) are treated as zero.
 *          If RCOND < 0, machine precision is used instead.
 *
 *  RANK    (output) INTEGER
 *          The effective rank of A, i.e., the number of singular values
 *          which are greater than RCOND*S(1).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= 1, and also:
 *          LWORK >= 3*min(M,N) + max( 2*min(M,N), max(M,N), NRHS )
 *          For good performance, LWORK should generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  the algorithm for computing the SVD failed to converge;
 *                if INFO = i, i off-diagonal elements of an intermediate
 *                bidiagonal form did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGELSS(int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank, double* work, int lwork)
{
    int info;
    ::F_DGELSS(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  This routine is deprecated and has been replaced by routine DGELSY.
 *
 *  DGELSX computes the minimum-norm solution to a real linear least
 *  squares problem:
 *      minimize || A * X - B ||
 *  using a complete orthogonal factorization of A.  A is an M-by-N
 *  matrix which may be rank-deficient.
 *
 *  Several right hand side vectors b and solution vectors x can be
 *  handled in a single call; they are stored as the columns of the
 *  M-by-NRHS right hand side matrix B and the N-by-NRHS solution
 *  matrix X.
 *
 *  The routine first computes a QR factorization with column pivoting:
 *      A * P = Q * [ R11 R12 ]
 *                  [  0  R22 ]
 *  with R11 defined as the largest leading submatrix whose estimated
 *  condition number is less than 1/RCOND.  The order of R11, RANK,
 *  is the effective rank of A.
 *
 *  Then, R22 is considered to be negligible, and R12 is annihilated
 *  by orthogonal transformations from the right, arriving at the
 *  complete orthogonal factorization:
 *     A * P = Q * [ T11 0 ] * Z
 *                 [  0  0 ]
 *  The minimum-norm solution is then
 *     X = P * Z' [ inv(T11)*Q1'*B ]
 *                [        0       ]
 *  where Q1 consists of the first RANK columns of Q.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of
 *          columns of matrices B and X. NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, A has been overwritten by details of its
 *          complete orthogonal factorization.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the M-by-NRHS right hand side matrix B.
 *          On exit, the N-by-NRHS solution matrix X.
 *          If m >= n and RANK = n, the residual sum-of-squares for
 *          the solution in the i-th column is given by the sum of
 *          squares of elements N+1:M in that column.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,M,N).
 *
 *  JPVT    (input/output) INTEGER array, dimension (N)
 *          On entry, if JPVT(i) .ne. 0, the i-th column of A is an
 *          initial column, otherwise it is a free column.  Before
 *          the QR factorization of A, all initial columns are
 *          permuted to the leading positions; only the remaining
 *          free columns are moved as a result of column pivoting
 *          during the factorization.
 *          On exit, if JPVT(i) = k, then the i-th column of A*P
 *          was the k-th column of A.
 *
 *  RCOND   (input) DOUBLE PRECISION
 *          RCOND is used to determine the effective rank of A, which
 *          is defined as the order of the largest leading triangular
 *          submatrix R11 in the QR factorization with pivoting of A,
 *          whose estimated condition number < 1/RCOND.
 *
 *  RANK    (output) INTEGER
 *          The effective rank of A, i.e., the order of the submatrix
 *          R11.  This is the same as the order of the submatrix T11
 *          in the complete orthogonal factorization of A.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension
 *                      (max( min(M,N)+3*N, 2*min(M,N)+NRHS )),
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGELSX(int m, int n, int nrhs, double* a, int lda, double* b, int ldb, int* jpvt, double rcond, int* rank, double* work)
{
    int info;
    ::F_DGELSX(&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGELSY computes the minimum-norm solution to a real linear least
 *  squares problem:
 *      minimize || A * X - B ||
 *  using a complete orthogonal factorization of A.  A is an M-by-N
 *  matrix which may be rank-deficient.
 *
 *  Several right hand side vectors b and solution vectors x can be
 *  handled in a single call; they are stored as the columns of the
 *  M-by-NRHS right hand side matrix B and the N-by-NRHS solution
 *  matrix X.
 *
 *  The routine first computes a QR factorization with column pivoting:
 *      A * P = Q * [ R11 R12 ]
 *                  [  0  R22 ]
 *  with R11 defined as the largest leading submatrix whose estimated
 *  condition number is less than 1/RCOND.  The order of R11, RANK,
 *  is the effective rank of A.
 *
 *  Then, R22 is considered to be negligible, and R12 is annihilated
 *  by orthogonal transformations from the right, arriving at the
 *  complete orthogonal factorization:
 *     A * P = Q * [ T11 0 ] * Z
 *                 [  0  0 ]
 *  The minimum-norm solution is then
 *     X = P * Z' [ inv(T11)*Q1'*B ]
 *                [        0       ]
 *  where Q1 consists of the first RANK columns of Q.
 *
 *  This routine is basically identical to the original xGELSX except
 *  three differences:
 *    o The call to the subroutine xGEQPF has been substituted by the
 *      the call to the subroutine xGEQP3. This subroutine is a Blas-3
 *      version of the QR factorization with column pivoting.
 *    o Matrix B (the right hand side) is updated with Blas-3.
 *    o The permutation of matrix B (the right hand side) is faster and
 *      more simple.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of
 *          columns of matrices B and X. NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, A has been overwritten by details of its
 *          complete orthogonal factorization.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the M-by-NRHS right hand side matrix B.
 *          On exit, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,M,N).
 *
 *  JPVT    (input/output) INTEGER array, dimension (N)
 *          On entry, if JPVT(i) .ne. 0, the i-th column of A is permuted
 *          to the front of AP, otherwise column i is a free column.
 *          On exit, if JPVT(i) = k, then the i-th column of AP
 *          was the k-th column of A.
 *
 *  RCOND   (input) DOUBLE PRECISION
 *          RCOND is used to determine the effective rank of A, which
 *          is defined as the order of the largest leading triangular
 *          submatrix R11 in the QR factorization with pivoting of A,
 *          whose estimated condition number < 1/RCOND.
 *
 *  RANK    (output) INTEGER
 *          The effective rank of A, i.e., the order of the submatrix
 *          R11.  This is the same as the order of the submatrix T11
 *          in the complete orthogonal factorization of A.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          The unblocked strategy requires that:
 *             LWORK >= MAX( MN+3*N+1, 2*MN+NRHS ),
 *          where MN = min( M, N ).
 *          The block algorithm requires that:
 *             LWORK >= MAX( MN+2*N+NB*(N+1), 2*MN+NB*NRHS ),
 *          where NB is an upper bound on the blocksize returned
 *          by ILAENV for the routines DGEQP3, DTZRZF, STZRQF, DORMQR,
 *          and DORMRZ.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: If INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA
 *    E. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain
 *    G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGELSY(int m, int n, int nrhs, double* a, int lda, double* b, int ldb, int* jpvt, double rcond, int* rank, double* work, int lwork)
{
    int info;
    ::F_DGELSY(&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEQLF computes a QL factorization of a real M-by-N matrix A:
 *  A = Q * L.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit,
 *          if m >= n, the lower triangle of the subarray
 *          A(m-n+1:m,1:n) contains the N-by-N lower triangular matrix L;
 *          if m <= n, the elements on and below the (n-m)-th
 *          superdiagonal contain the M-by-N lower trapezoidal matrix L;
 *          the remaining elements, with the array TAU, represent the
 *          orthogonal matrix Q as a product of elementary reflectors
 *          (see Further Details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors (see Further
 *          Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,N).
 *          For optimum performance LWORK >= N*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(k) . . . H(2) H(1), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(m-k+i+1:m) = 0 and v(m-k+i) = 1; v(1:m-k+i-1) is stored on exit in
 *  A(1:m-k+i-1,n-k+i), and tau in TAU(i).
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGEQLF(int m, int n, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DGEQLF(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEQP3 computes a QR factorization with column pivoting of a
 *  matrix A:  A*P = Q*R  using Level 3 BLAS.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, the upper triangle of the array contains the
 *          min(M,N)-by-N upper trapezoidal matrix R; the elements below
 *          the diagonal, together with the array TAU, represent the
 *          orthogonal matrix Q as a product of min(M,N) elementary
 *          reflectors.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  JPVT    (input/output) INTEGER array, dimension (N)
 *          On entry, if JPVT(J).ne.0, the J-th column of A is permuted
 *          to the front of A*P (a leading column); if JPVT(J)=0,
 *          the J-th column of A is a free column.
 *          On exit, if JPVT(J)=K, then the J-th column of A*P was the
 *          the K-th column of A.
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO=0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= 3*N+1.
 *          For optimal performance LWORK >= 2*N+( N+1 )*NB, where NB
 *          is the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit.
 *          < 0: if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real/complex scalar, and v is a real/complex vector
 *  with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in
 *  A(i+1:m,i), and tau in TAU(i).
 *
 *  Based on contributions by
 *    G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain
 *    X. Sun, Computer Science Dept., Duke University, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEQP3(int m, int n, double* a, int lda, int* jpvt, double* tau, double* work, int lwork)
{
    int info;
    ::F_DGEQP3(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  This routine is deprecated and has been replaced by routine DGEQP3.
 *
 *  DGEQPF computes a QR factorization with column pivoting of a
 *  real M-by-N matrix A: A*P = Q*R.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A. N >= 0
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, the upper triangle of the array contains the
 *          min(M,N)-by-N upper triangular matrix R; the elements
 *          below the diagonal, together with the array TAU,
 *          represent the orthogonal matrix Q as a product of
 *          min(m,n) elementary reflectors.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  JPVT    (input/output) INTEGER array, dimension (N)
 *          On entry, if JPVT(i) .ne. 0, the i-th column of A is permuted
 *          to the front of A*P (a leading column); if JPVT(i) = 0,
 *          the i-th column of A is a free column.
 *          On exit, if JPVT(i) = k, then the i-th column of A*P
 *          was the k-th column of A.
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(n)
 *
 *  Each H(i) has the form
 *
 *     H = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i).
 *
 *  The matrix P is represented in jpvt as follows: If
 *     jpvt(j) = i
 *  then the jth column of P is the ith canonical unit vector.
 *
 *  Partial column norm updating strategy modified by
 *    Z. Drmac and Z. Bujanovic, Dept. of Mathematics,
 *    University of Zagreb, Croatia.
 *     June 2010
 *  For more details see LAPACK Working Note 176.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGEQPF(int m, int n, double* a, int lda, int* jpvt, double* tau, double* work)
{
    int info;
    ::F_DGEQPF(&m, &n, a, &lda, jpvt, tau, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGEQRF computes a QR factorization of a real M-by-N matrix A:
 *  A = Q * R.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array
 *          contain the min(M,N)-by-N upper trapezoidal matrix R (R is
 *          upper triangular if m >= n); the elements below the diagonal,
 *          with the array TAU, represent the orthogonal matrix Q as a
 *          product of min(m,n) elementary reflectors (see Further
 *          Details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors (see Further
 *          Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,N).
 *          For optimum performance LWORK >= N*NB, where NB is
 *          the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
 *  and tau in TAU(i).
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGEQRF(int m, int n, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DGEQRF(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGERFS improves the computed solution to a system of linear
 *  equations and provides error bounds and backward error estimates for
 *  the solution.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B     (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The original N-by-N matrix A.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  AF      (input) DOUBLE PRECISION array, dimension (LDAF,N)
 *          The factors L and U from the factorization A = P*L*U
 *          as computed by DGETRF.
 *
 *  LDAF    (input) INTEGER
 *          The leading dimension of the array AF.  LDAF >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices from DGETRF; for 1<=i<=N, row i of the
 *          matrix was interchanged with row IPIV(i).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DGETRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGERFS(char trans, int n, int nrhs, double* a, int lda, double* af, int ldaf, int* ipiv, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DGERFS(&trans, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGERQF computes an RQ factorization of a real M-by-N matrix A:
 *  A = R * Q.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit,
 *          if m <= n, the upper triangle of the subarray
 *          A(1:m,n-m+1:n) contains the M-by-M upper triangular matrix R;
 *          if m >= n, the elements on and above the (m-n)-th subdiagonal
 *          contain the M-by-N upper trapezoidal matrix R;
 *          the remaining elements, with the array TAU, represent the
 *          orthogonal matrix Q as a product of min(m,n) elementary
 *          reflectors (see Further Details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors (see Further
 *          Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,M).
 *          For optimum performance LWORK >= M*NB, where NB is
 *          the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(n-k+i+1:n) = 0 and v(n-k+i) = 1; v(1:n-k+i-1) is stored on exit in
 *  A(m-k+i,1:n-k+i-1), and tau in TAU(i).
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGERQF(int m, int n, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DGERQF(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGESVX uses the LU factorization to compute the solution to a real
 *  system of linear equations
 *     A * X = B,
 *  where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'E', real scaling factors are computed to equilibrate
 *     the system:
 *        TRANS = 'N':  diag(R)*A*diag(C)     *inv(diag(C))*X = diag(R)*B
 *        TRANS = 'T': (diag(R)*A*diag(C))**T *inv(diag(R))*X = diag(C)*B
 *        TRANS = 'C': (diag(R)*A*diag(C))**H *inv(diag(R))*X = diag(C)*B
 *     Whether or not the system will be equilibrated depends on the
 *     scaling of the matrix A, but if equilibration is used, A is
 *     overwritten by diag(R)*A*diag(C) and B by diag(R)*B (if TRANS='N')
 *     or diag(C)*B (if TRANS = 'T' or 'C').
 *
 *  2. If FACT = 'N' or 'E', the LU decomposition is used to factor the
 *     matrix A (after equilibration if FACT = 'E') as
 *        A = P * L * U,
 *     where P is a permutation matrix, L is a unit lower triangular
 *     matrix, and U is upper triangular.
 *
 *  3. If some U(i,i)=0, so that U is exactly singular, then the routine
 *     returns with INFO = i. Otherwise, the factored form of A is used
 *     to estimate the condition number of the matrix A.  If the
 *     reciprocal of the condition number is less than machine precision,
 *  C++ Return value: INFO    (output) INTEGER
 *     to solve for X and compute error bounds as described below.
 *
 *  4. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  5. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  6. If equilibration was used, the matrix X is premultiplied by
 *     diag(C) (if TRANS = 'N') or diag(R) (if TRANS = 'T' or 'C') so
 *     that it solves the original system before equilibration.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of the matrix A is
 *          supplied on entry, and if not, whether the matrix A should be
 *          equilibrated before it is factored.
 *          = 'F':  On entry, AF and IPIV contain the factored form of A.
 *                  If EQUED is not 'N', the matrix A has been
 *                  equilibrated with scaling factors given by R and C.
 *                  A, AF, and IPIV are not modified.
 *          = 'N':  The matrix A will be copied to AF and factored.
 *          = 'E':  The matrix A will be equilibrated if necessary, then
 *                  copied to AF and factored.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B     (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Transpose)
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the N-by-N matrix A.  If FACT = 'F' and EQUED is
 *          not 'N', then A must have been equilibrated by the scaling
 *          factors in R and/or C.  A is not modified if FACT = 'F' or
 *          'N', or if FACT = 'E' and EQUED = 'N' on exit.
 *
 *          On exit, if EQUED .ne. 'N', A is scaled as follows:
 *          EQUED = 'R':  A := diag(R) * A
 *          EQUED = 'C':  A := A * diag(C)
 *          EQUED = 'B':  A := diag(R) * A * diag(C).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  AF      (input or output) DOUBLE PRECISION array, dimension (LDAF,N)
 *          If FACT = 'F', then AF is an input argument and on entry
 *          contains the factors L and U from the factorization
 *          A = P*L*U as computed by DGETRF.  If EQUED .ne. 'N', then
 *          AF is the factored form of the equilibrated matrix A.
 *
 *          If FACT = 'N', then AF is an output argument and on exit
 *          returns the factors L and U from the factorization A = P*L*U
 *          of the original matrix A.
 *
 *          If FACT = 'E', then AF is an output argument and on exit
 *          returns the factors L and U from the factorization A = P*L*U
 *          of the equilibrated matrix A (see the description of A for
 *          the form of the equilibrated matrix).
 *
 *  LDAF    (input) INTEGER
 *          The leading dimension of the array AF.  LDAF >= max(1,N).
 *
 *  IPIV    (input or output) INTEGER array, dimension (N)
 *          If FACT = 'F', then IPIV is an input argument and on entry
 *          contains the pivot indices from the factorization A = P*L*U
 *          as computed by DGETRF; row i of the matrix was interchanged
 *          with row IPIV(i).
 *
 *          If FACT = 'N', then IPIV is an output argument and on exit
 *          contains the pivot indices from the factorization A = P*L*U
 *          of the original matrix A.
 *
 *          If FACT = 'E', then IPIV is an output argument and on exit
 *          contains the pivot indices from the factorization A = P*L*U
 *          of the equilibrated matrix A.
 *
 *  EQUED   (input or output) CHARACTER*1
 *          Specifies the form of equilibration that was done.
 *          = 'N':  No equilibration (always true if FACT = 'N').
 *          = 'R':  Row equilibration, i.e., A has been premultiplied by
 *                  diag(R).
 *          = 'C':  Column equilibration, i.e., A has been postmultiplied
 *                  by diag(C).
 *          = 'B':  Both row and column equilibration, i.e., A has been
 *                  replaced by diag(R) * A * diag(C).
 *          EQUED is an input argument if FACT = 'F'; otherwise, it is an
 *          output argument.
 *
 *  R       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The row scale factors for A.  If EQUED = 'R' or 'B', A is
 *          multiplied on the left by diag(R); if EQUED = 'N' or 'C', R
 *          is not accessed.  R is an input argument if FACT = 'F';
 *          otherwise, R is an output argument.  If FACT = 'F' and
 *          EQUED = 'R' or 'B', each element of R must be positive.
 *
 *  C       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The column scale factors for A.  If EQUED = 'C' or 'B', A is
 *          multiplied on the right by diag(C); if EQUED = 'N' or 'R', C
 *          is not accessed.  C is an input argument if FACT = 'F';
 *          otherwise, C is an output argument.  If FACT = 'F' and
 *          EQUED = 'C' or 'B', each element of C must be positive.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit,
 *          if EQUED = 'N', B is not modified;
 *          if TRANS = 'N' and EQUED = 'R' or 'B', B is overwritten by
 *          diag(R)*B;
 *          if TRANS = 'T' or 'C' and EQUED = 'C' or 'B', B is
 *          overwritten by diag(C)*B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X
 *          to the original system of equations.  Note that A and B are
 *          modified on exit if EQUED .ne. 'N', and the solution to the
 *          equilibrated system is inv(diag(C))*X if TRANS = 'N' and
 *          EQUED = 'C' or 'B', or inv(diag(R))*X if TRANS = 'T' or 'C'
 *          and EQUED = 'R' or 'B'.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A after equilibration (if done).  If RCOND is less than the
 *          machine precision (in particular, if RCOND = 0), the matrix
 *          is singular to working precision.  This condition is
 *          indicated by a return code of INFO > 0.
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (4*N)
 *          On exit, WORK(1) contains the reciprocal pivot growth
 *          factor norm(A)/norm(U). The "max absolute element" norm is
 *          used. If WORK(1) is much less than 1, then the stability
 *          of the LU factorization of the (equilibrated) matrix A
 *          could be poor. This also means that the solution X, condition
 *          estimator RCOND, and forward error bound FERR could be
 *          unreliable. If factorization fails with 0<INFO<=N, then
 *          WORK(1) contains the reciprocal pivot growth factor for the
 *          leading INFO columns of A.
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  U(i,i) is exactly zero.  The factorization has
 *                       been completed, but the factor U is exactly
 *                       singular, so the solution and error bounds
 *                       could not be computed. RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGESVX(char fact, char trans, int n, int nrhs, double* a, int lda, double* af, int ldaf, int* ipiv, char equed, double* r, double* c, double* b, int ldb, double* x, int ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DGESVX(&fact, &trans, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, r, c, b, &ldb, x, &ldx, rcond, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGETRF computes an LU factorization of a general M-by-N matrix A
 *  using partial pivoting with row interchanges.
 *
 *  The factorization has the form
 *     A = P * L * U
 *  where P is a permutation matrix, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 *
 *  This is the right-looking Level 3 BLAS version of the algorithm.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix to be factored.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  IPIV    (output) INTEGER array, dimension (min(M,N))
 *          The pivot indices; for 1 <= i <= min(M,N), row i of the
 *          matrix was interchanged with row IPIV(i).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
 *                has been completed, but the factor U is exactly
 *                singular, and division by zero will occur if it is used
 *                to solve a system of equations.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGETRF(int m, int n, double* a, int lda, int* ipiv)
{
    int info;
    ::F_DGETRF(&m, &n, a, &lda, ipiv, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGETRI computes the inverse of a matrix using the LU factorization
 *  computed by DGETRF.
 *
 *  This method inverts U and then computes inv(A) by solving the system
 *  inv(A)*L = inv(U) for inv(A).
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the factors L and U from the factorization
 *          A = P*L*U as computed by DGETRF.
 *          On exit, if INFO = 0, the inverse of the original matrix A.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices from DGETRF; for 1<=i<=N, row i of the
 *          matrix was interchanged with row IPIV(i).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO=0, then WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,N).
 *          For optimal performance LWORK >= N*NB, where NB is
 *          the optimal blocksize returned by ILAENV.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, U(i,i) is exactly zero; the matrix is
 *                singular and its inverse could not be computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGETRI(int n, double* a, int lda, int* ipiv, double* work, int lwork)
{
    int info;
    ::F_DGETRI(&n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGETRS solves a system of linear equations
 *     A * X = B  or  A' * X = B
 *  with a general N-by-N matrix A using the LU factorization computed
 *  by DGETRF.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A'* X = B  (Transpose)
 *          = 'C':  A'* X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The factors L and U from the factorization A = P*L*U
 *          as computed by DGETRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices from DGETRF; for 1<=i<=N, row i of the
 *          matrix was interchanged with row IPIV(i).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGETRS(char trans, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DGETRS(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGBAK forms the right or left eigenvectors of a real generalized
 *  eigenvalue problem A*x = lambda*B*x, by backward transformation on
 *  the computed eigenvectors of the balanced pair of matrices output by
 *  DGGBAL.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies the type of backward transformation required:
 *          = 'N':  do nothing, return immediately;
 *          = 'P':  do backward transformation for permutation only;
 *          = 'S':  do backward transformation for scaling only;
 *          = 'B':  do backward transformations for both permutation and
 *                  scaling.
 *          JOB must be the same as the argument JOB supplied to DGGBAL.
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'R':  V contains right eigenvectors;
 *          = 'L':  V contains left eigenvectors.
 *
 *  N       (input) INTEGER
 *          The number of rows of the matrix V.  N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          The integers ILO and IHI determined by DGGBAL.
 *          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
 *
 *  LSCALE  (input) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and/or scaling factors applied
 *          to the left side of A and B, as returned by DGGBAL.
 *
 *  RSCALE  (input) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and/or scaling factors applied
 *          to the right side of A and B, as returned by DGGBAL.
 *
 *  M       (input) INTEGER
 *          The number of columns of the matrix V.  M >= 0.
 *
 *  V       (input/output) DOUBLE PRECISION array, dimension (LDV,M)
 *          On entry, the matrix of right or left eigenvectors to be
 *          transformed, as returned by DTGEVC.
 *          On exit, V is overwritten by the transformed eigenvectors.
 *
 *  LDV     (input) INTEGER
 *          The leading dimension of the matrix V. LDV >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  See R.C. Ward, Balancing the generalized eigenvalue problem,
 *                 SIAM J. Sci. Stat. Comp. 2 (1981), 141-152.
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGGBAK(char job, char side, int n, int ilo, int ihi, double* lscale, double* rscale, int m, double* v, int ldv)
{
    int info;
    ::F_DGGBAK(&job, &side, &n, &ilo, &ihi, lscale, rscale, &m, v, &ldv, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGBAL balances a pair of general real matrices (A,B).  This
 *  involves, first, permuting A and B by similarity transformations to
 *  isolate eigenvalues in the first 1 to ILO$-$1 and last IHI+1 to N
 *  elements on the diagonal; and second, applying a diagonal similarity
 *  transformation to rows and columns ILO to IHI to make the rows
 *  and columns as close in norm as possible. Both steps are optional.
 *
 *  Balancing may reduce the 1-norm of the matrices, and improve the
 *  accuracy of the computed eigenvalues and/or eigenvectors in the
 *  generalized eigenvalue problem A*x = lambda*B*x.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies the operations to be performed on A and B:
 *          = 'N':  none:  simply set ILO = 1, IHI = N, LSCALE(I) = 1.0
 *                  and RSCALE(I) = 1.0 for i = 1,...,N.
 *          = 'P':  permute only;
 *          = 'S':  scale only;
 *          = 'B':  both permute and scale.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the input matrix A.
 *          On exit,  A is overwritten by the balanced matrix.
 *          If JOB = 'N', A is not referenced.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the input matrix B.
 *          On exit,  B is overwritten by the balanced matrix.
 *          If JOB = 'N', B is not referenced.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  ILO     (output) INTEGER
 *  IHI     (output) INTEGER
 *          ILO and IHI are set to integers such that on exit
 *          A(i,j) = 0 and B(i,j) = 0 if i > j and
 *          j = 1,...,ILO-1 or i = IHI+1,...,N.
 *          If JOB = 'N' or 'S', ILO = 1 and IHI = N.
 *
 *  LSCALE  (output) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and scaling factors applied
 *          to the left side of A and B.  If P(j) is the index of the
 *          row interchanged with row j, and D(j)
 *          is the scaling factor applied to row j, then
 *            LSCALE(j) = P(j)    for J = 1,...,ILO-1
 *                      = D(j)    for J = ILO,...,IHI
 *                      = P(j)    for J = IHI+1,...,N.
 *          The order in which the interchanges are made is N to IHI+1,
 *          then 1 to ILO-1.
 *
 *  RSCALE  (output) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and scaling factors applied
 *          to the right side of A and B.  If P(j) is the index of the
 *          column interchanged with column j, and D(j)
 *          is the scaling factor applied to column j, then
 *            LSCALE(j) = P(j)    for J = 1,...,ILO-1
 *                      = D(j)    for J = ILO,...,IHI
 *                      = P(j)    for J = IHI+1,...,N.
 *          The order in which the interchanges are made is N to IHI+1,
 *          then 1 to ILO-1.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (lwork)
 *          lwork must be at least max(1,6*N) when JOB = 'S' or 'B', and
 *          at least 1 when JOB = 'N' or 'P'.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  See R.C. WARD, Balancing the generalized eigenvalue problem,
 *                 SIAM J. Sci. Stat. Comp. 2 (1981), 141-152.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGBAL(char job, int n, double* a, int lda, double* b, int ldb, int* ilo, int* ihi, double* lscale, double* rscale, double* work)
{
    int info;
    ::F_DGGBAL(&job, &n, a, &lda, b, &ldb, ilo, ihi, lscale, rscale, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGES computes for a pair of N-by-N real nonsymmetric matrices (A,B),
 *  the generalized eigenvalues, the generalized real Schur form (S,T),
 *  optionally, the left and/or right matrices of Schur vectors (VSL and
 *  VSR). This gives the generalized Schur factorization
 *
 *           (A,B) = ( (VSL)*S*(VSR)**T, (VSL)*T*(VSR)**T )
 *
 *  Optionally, it also orders the eigenvalues so that a selected cluster
 *  of eigenvalues appears in the leading diagonal blocks of the upper
 *  quasi-triangular matrix S and the upper triangular matrix T.The
 *  leading columns of VSL and VSR then form an orthonormal basis for the
 *  corresponding left and right eigenspaces (deflating subspaces).
 *
 *  (If only the generalized eigenvalues are needed, use the driver
 *  DGGEV instead, which is faster.)
 *
 *  A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
 *  or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
 *  usually represented as the pair (alpha,beta), as there is a
 *  reasonable interpretation for beta=0 or both being zero.
 *
 *  A pair of matrices (S,T) is in generalized real Schur form if T is
 *  upper triangular with non-negative diagonal and S is block upper
 *  triangular with 1-by-1 and 2-by-2 blocks.  1-by-1 blocks correspond
 *  to real generalized eigenvalues, while 2-by-2 blocks of S will be
 *  "standardized" by making the corresponding elements of T have the
 *  form:
 *          [  a  0  ]
 *          [  0  b  ]
 *
 *  and the pair of corresponding 2-by-2 blocks in S and T will have a
 *  complex conjugate pair of generalized eigenvalues.
 *
 *
 *  Arguments
 *  =========
 *
 *  JOBVSL  (input) CHARACTER*1
 *          = 'N':  do not compute the left Schur vectors;
 *          = 'V':  compute the left Schur vectors.
 *
 *  JOBVSR  (input) CHARACTER*1
 *          = 'N':  do not compute the right Schur vectors;
 *          = 'V':  compute the right Schur vectors.
 *
 *  SORT    (input) CHARACTER*1
 *          Specifies whether or not to order the eigenvalues on the
 *          diagonal of the generalized Schur form.
 *          = 'N':  Eigenvalues are not ordered;
 *          = 'S':  Eigenvalues are ordered (see SELCTG);
 *
 *  SELCTG  (external procedure) LOGICAL FUNCTION of three DOUBLE PRECISION arguments
 *          SELCTG must be declared EXTERNAL in the calling subroutine.
 *          If SORT = 'N', SELCTG is not referenced.
 *          If SORT = 'S', SELCTG is used to select eigenvalues to sort
 *          to the top left of the Schur form.
 *          An eigenvalue (ALPHAR(j)+ALPHAI(j))/BETA(j) is selected if
 *          SELCTG(ALPHAR(j),ALPHAI(j),BETA(j)) is true; i.e. if either
 *          one of a complex conjugate pair of eigenvalues is selected,
 *          then both complex eigenvalues are selected.
 *
 *          Note that in the ill-conditioned case, a selected complex
 *          eigenvalue may no longer satisfy SELCTG(ALPHAR(j),ALPHAI(j),
 *          BETA(j)) = .TRUE. after ordering. INFO is to be set to N+2
 *          in this case.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A, B, VSL, and VSR.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the first of the pair of matrices.
 *          On exit, A has been overwritten by its generalized Schur
 *          form S.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the second of the pair of matrices.
 *          On exit, B has been overwritten by its generalized Schur
 *          form T.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of B.  LDB >= max(1,N).
 *
 *  SDIM    (output) INTEGER
 *          If SORT = 'N', SDIM = 0.
 *          If SORT = 'S', SDIM = number of eigenvalues (after sorting)
 *          for which SELCTG is true.  (Complex conjugate pairs for which
 *          SELCTG is true for either eigenvalue count as 2.)
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will
 *          be the generalized eigenvalues.  ALPHAR(j) + ALPHAI(j)*i,
 *          and  BETA(j),j=1,...,N are the diagonals of the complex Schur
 *          form (S,T) that would result if the 2-by-2 diagonal blocks of
 *          the real Schur form of (A,B) were further reduced to
 *          triangular form using 2-by-2 complex unitary transformations.
 *          If ALPHAI(j) is zero, then the j-th eigenvalue is real; if
 *          positive, then the j-th and (j+1)-st eigenvalues are a
 *          complex conjugate pair, with ALPHAI(j+1) negative.
 *
 *          Note: the quotients ALPHAR(j)/BETA(j) and ALPHAI(j)/BETA(j)
 *          may easily over- or underflow, and BETA(j) may even be zero.
 *          Thus, the user should avoid naively computing the ratio.
 *          However, ALPHAR and ALPHAI will be always less than and
 *          usually comparable with norm(A) in magnitude, and BETA always
 *          less than and usually comparable with norm(B).
 *
 *  VSL     (output) DOUBLE PRECISION array, dimension (LDVSL,N)
 *          If JOBVSL = 'V', VSL will contain the left Schur vectors.
 *          Not referenced if JOBVSL = 'N'.
 *
 *  LDVSL   (input) INTEGER
 *          The leading dimension of the matrix VSL. LDVSL >=1, and
 *          if JOBVSL = 'V', LDVSL >= N.
 *
 *  VSR     (output) DOUBLE PRECISION array, dimension (LDVSR,N)
 *          If JOBVSR = 'V', VSR will contain the right Schur vectors.
 *          Not referenced if JOBVSR = 'N'.
 *
 *  LDVSR   (input) INTEGER
 *          The leading dimension of the matrix VSR. LDVSR >= 1, and
 *          if JOBVSR = 'V', LDVSR >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If N = 0, LWORK >= 1, else LWORK >= 8*N+16.
 *          For good performance , LWORK must generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  BWORK   (workspace) LOGICAL array, dimension (N)
 *          Not referenced if SORT = 'N'.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1,...,N:
 *                The QZ iteration failed.  (A,B) are not in Schur
 *                form, but ALPHAR(j), ALPHAI(j), and BETA(j) should
 *                be correct for j=INFO+1,...,N.
 *          > N:  =N+1: other than QZ iteration failed in DHGEQZ.
 *                =N+2: after reordering, roundoff changed values of
 *                      some complex eigenvalues so that leading
 *                      eigenvalues in the Generalized Schur form no
 *                      longer satisfy SELCTG=.TRUE.  This could also
 *                      be caused due to scaling.
 *                =N+3: reordering failed in DTGSEN.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGES(char jobvsl, char jobvsr, char sort, int n, double* a, int lda, double* b, int ldb, int* sdim, double* alphar, double* alphai, double* beta, double* vsl, int ldvsl, double* vsr, int ldvsr, double* work, int lwork)
{
    int info;
    ::F_DGGES(&jobvsl, &jobvsr, &sort, &n, a, &lda, b, &ldb, sdim, alphar, alphai, beta, vsl, &ldvsl, vsr, &ldvsr, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGESX computes for a pair of N-by-N real nonsymmetric matrices
 *  (A,B), the generalized eigenvalues, the real Schur form (S,T), and,
 *  optionally, the left and/or right matrices of Schur vectors (VSL and
 *  VSR).  This gives the generalized Schur factorization
 *
 *       (A,B) = ( (VSL) S (VSR)**T, (VSL) T (VSR)**T )
 *
 *  Optionally, it also orders the eigenvalues so that a selected cluster
 *  of eigenvalues appears in the leading diagonal blocks of the upper
 *  quasi-triangular matrix S and the upper triangular matrix T; computes
 *  a reciprocal condition number for the average of the selected
 *  eigenvalues (RCONDE); and computes a reciprocal condition number for
 *  the right and left deflating subspaces corresponding to the selected
 *  eigenvalues (RCONDV). The leading columns of VSL and VSR then form
 *  an orthonormal basis for the corresponding left and right eigenspaces
 *  (deflating subspaces).
 *
 *  A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
 *  or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
 *  usually represented as the pair (alpha,beta), as there is a
 *  reasonable interpretation for beta=0 or for both being zero.
 *
 *  A pair of matrices (S,T) is in generalized real Schur form if T is
 *  upper triangular with non-negative diagonal and S is block upper
 *  triangular with 1-by-1 and 2-by-2 blocks.  1-by-1 blocks correspond
 *  to real generalized eigenvalues, while 2-by-2 blocks of S will be
 *  "standardized" by making the corresponding elements of T have the
 *  form:
 *          [  a  0  ]
 *          [  0  b  ]
 *
 *  and the pair of corresponding 2-by-2 blocks in S and T will have a
 *  complex conjugate pair of generalized eigenvalues.
 *
 *
 *  Arguments
 *  =========
 *
 *  JOBVSL  (input) CHARACTER*1
 *          = 'N':  do not compute the left Schur vectors;
 *          = 'V':  compute the left Schur vectors.
 *
 *  JOBVSR  (input) CHARACTER*1
 *          = 'N':  do not compute the right Schur vectors;
 *          = 'V':  compute the right Schur vectors.
 *
 *  SORT    (input) CHARACTER*1
 *          Specifies whether or not to order the eigenvalues on the
 *          diagonal of the generalized Schur form.
 *          = 'N':  Eigenvalues are not ordered;
 *          = 'S':  Eigenvalues are ordered (see SELCTG).
 *
 *  SELCTG  (external procedure) LOGICAL FUNCTION of three DOUBLE PRECISION arguments
 *          SELCTG must be declared EXTERNAL in the calling subroutine.
 *          If SORT = 'N', SELCTG is not referenced.
 *          If SORT = 'S', SELCTG is used to select eigenvalues to sort
 *          to the top left of the Schur form.
 *          An eigenvalue (ALPHAR(j)+ALPHAI(j))/BETA(j) is selected if
 *          SELCTG(ALPHAR(j),ALPHAI(j),BETA(j)) is true; i.e. if either
 *          one of a complex conjugate pair of eigenvalues is selected,
 *          then both complex eigenvalues are selected.
 *          Note that a selected complex eigenvalue may no longer satisfy
 *          SELCTG(ALPHAR(j),ALPHAI(j),BETA(j)) = .TRUE. after ordering,
 *          since ordering may change the value of complex eigenvalues
 *          (especially if the eigenvalue is ill-conditioned), in this
 *          case INFO is set to N+3.
 *
 *  SENSE   (input) CHARACTER*1
 *          Determines which reciprocal condition numbers are computed.
 *          = 'N' : None are computed;
 *          = 'E' : Computed for average of selected eigenvalues only;
 *          = 'V' : Computed for selected deflating subspaces only;
 *          = 'B' : Computed for both.
 *          If SENSE = 'E', 'V', or 'B', SORT must equal 'S'.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A, B, VSL, and VSR.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the first of the pair of matrices.
 *          On exit, A has been overwritten by its generalized Schur
 *          form S.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the second of the pair of matrices.
 *          On exit, B has been overwritten by its generalized Schur
 *          form T.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of B.  LDB >= max(1,N).
 *
 *  SDIM    (output) INTEGER
 *          If SORT = 'N', SDIM = 0.
 *          If SORT = 'S', SDIM = number of eigenvalues (after sorting)
 *          for which SELCTG is true.  (Complex conjugate pairs for which
 *          SELCTG is true for either eigenvalue count as 2.)
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will
 *          be the generalized eigenvalues.  ALPHAR(j) + ALPHAI(j)*i
 *          and BETA(j),j=1,...,N  are the diagonals of the complex Schur
 *          form (S,T) that would result if the 2-by-2 diagonal blocks of
 *          the real Schur form of (A,B) were further reduced to
 *          triangular form using 2-by-2 complex unitary transformations.
 *          If ALPHAI(j) is zero, then the j-th eigenvalue is real; if
 *          positive, then the j-th and (j+1)-st eigenvalues are a
 *          complex conjugate pair, with ALPHAI(j+1) negative.
 *
 *          Note: the quotients ALPHAR(j)/BETA(j) and ALPHAI(j)/BETA(j)
 *          may easily over- or underflow, and BETA(j) may even be zero.
 *          Thus, the user should avoid naively computing the ratio.
 *          However, ALPHAR and ALPHAI will be always less than and
 *          usually comparable with norm(A) in magnitude, and BETA always
 *          less than and usually comparable with norm(B).
 *
 *  VSL     (output) DOUBLE PRECISION array, dimension (LDVSL,N)
 *          If JOBVSL = 'V', VSL will contain the left Schur vectors.
 *          Not referenced if JOBVSL = 'N'.
 *
 *  LDVSL   (input) INTEGER
 *          The leading dimension of the matrix VSL. LDVSL >=1, and
 *          if JOBVSL = 'V', LDVSL >= N.
 *
 *  VSR     (output) DOUBLE PRECISION array, dimension (LDVSR,N)
 *          If JOBVSR = 'V', VSR will contain the right Schur vectors.
 *          Not referenced if JOBVSR = 'N'.
 *
 *  LDVSR   (input) INTEGER
 *          The leading dimension of the matrix VSR. LDVSR >= 1, and
 *          if JOBVSR = 'V', LDVSR >= N.
 *
 *  RCONDE  (output) DOUBLE PRECISION array, dimension ( 2 )
 *          If SENSE = 'E' or 'B', RCONDE(1) and RCONDE(2) contain the
 *          reciprocal condition numbers for the average of the selected
 *          eigenvalues.
 *          Not referenced if SENSE = 'N' or 'V'.
 *
 *  RCONDV  (output) DOUBLE PRECISION array, dimension ( 2 )
 *          If SENSE = 'V' or 'B', RCONDV(1) and RCONDV(2) contain the
 *          reciprocal condition numbers for the selected deflating
 *          subspaces.
 *          Not referenced if SENSE = 'N' or 'E'.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If N = 0, LWORK >= 1, else if SENSE = 'E', 'V', or 'B',
 *          LWORK >= max( 8*N, 6*N+16, 2*SDIM*(N-SDIM) ), else
 *          LWORK >= max( 8*N, 6*N+16 ).
 *          Note that 2*SDIM*(N-SDIM) <= N*N/2.
 *          Note also that an error is only returned if
 *          LWORK < max( 8*N, 6*N+16), but if SENSE = 'E' or 'V' or 'B'
 *          this may not be large enough.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the bound on the optimal size of the WORK
 *          array and the minimum size of the IWORK array, returns these
 *          values as the first entries of the WORK and IWORK arrays, and
 *          no error message related to LWORK or LIWORK is issued by
 *          XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the minimum LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If SENSE = 'N' or N = 0, LIWORK >= 1, otherwise
 *          LIWORK >= N+6.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the bound on the optimal size of the
 *          WORK array and the minimum size of the IWORK array, returns
 *          these values as the first entries of the WORK and IWORK
 *          arrays, and no error message related to LWORK or LIWORK is
 *          issued by XERBLA.
 *
 *  BWORK   (workspace) LOGICAL array, dimension (N)
 *          Not referenced if SORT = 'N'.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1,...,N:
 *                The QZ iteration failed.  (A,B) are not in Schur
 *                form, but ALPHAR(j), ALPHAI(j), and BETA(j) should
 *                be correct for j=INFO+1,...,N.
 *          > N:  =N+1: other than QZ iteration failed in DHGEQZ
 *                =N+2: after reordering, roundoff changed values of
 *                      some complex eigenvalues so that leading
 *                      eigenvalues in the Generalized Schur form no
 *                      longer satisfy SELCTG=.TRUE.  This could also
 *                      be caused due to scaling.
 *                =N+3: reordering failed in DTGSEN.
 *
 *  Further Details
 *  ===============
 *
 *  An approximate (asymptotic) bound on the average absolute error of
 *  the selected eigenvalues is
 *
 *       EPS * norm((A, B)) / RCONDE( 1 ).
 *
 *  An approximate (asymptotic) bound on the maximum angular error in
 *  the computed deflating subspaces is
 *
 *       EPS * norm((A, B)) / RCONDV( 2 ).
 *
 *  See LAPACK User's Guide, section 4.11 for more information.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGESX(char jobvsl, char jobvsr, char sort, char sense, int n, double* a, int lda, double* b, int ldb, int* sdim, double* alphar, double* alphai, double* beta, double* vsl, int ldvsl, double* vsr, int ldvsr, double* rconde, double* rcondv, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DGGESX(&jobvsl, &jobvsr, &sort, &sense, &n, a, &lda, b, &ldb, sdim, alphar, alphai, beta, vsl, &ldvsl, vsr, &ldvsr, rconde, rcondv, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGEV computes for a pair of N-by-N real nonsymmetric matrices (A,B)
 *  the generalized eigenvalues, and optionally, the left and/or right
 *  generalized eigenvectors.
 *
 *  A generalized eigenvalue for a pair of matrices (A,B) is a scalar
 *  lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
 *  singular. It is usually represented as the pair (alpha,beta), as
 *  there is a reasonable interpretation for beta=0, and even for both
 *  being zero.
 *
 *  The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
 *  of (A,B) satisfies
 *
 *                   A * v(j) = lambda(j) * B * v(j).
 *
 *  The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
 *  of (A,B) satisfies
 *
 *                   u(j)**H * A  = lambda(j) * u(j)**H * B .
 *
 *  where u(j)**H is the conjugate-transpose of u(j).
 *
 *
 *  Arguments
 *  =========
 *
 *  JOBVL   (input) CHARACTER*1
 *          = 'N':  do not compute the left generalized eigenvectors;
 *          = 'V':  compute the left generalized eigenvectors.
 *
 *  JOBVR   (input) CHARACTER*1
 *          = 'N':  do not compute the right generalized eigenvectors;
 *          = 'V':  compute the right generalized eigenvectors.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A, B, VL, and VR.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the matrix A in the pair (A,B).
 *          On exit, A has been overwritten.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the matrix B in the pair (A,B).
 *          On exit, B has been overwritten.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of B.  LDB >= max(1,N).
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will
 *          be the generalized eigenvalues.  If ALPHAI(j) is zero, then
 *          the j-th eigenvalue is real; if positive, then the j-th and
 *          (j+1)-st eigenvalues are a complex conjugate pair, with
 *          ALPHAI(j+1) negative.
 *
 *          Note: the quotients ALPHAR(j)/BETA(j) and ALPHAI(j)/BETA(j)
 *          may easily over- or underflow, and BETA(j) may even be zero.
 *          Thus, the user should avoid naively computing the ratio
 *          alpha/beta.  However, ALPHAR and ALPHAI will be always less
 *          than and usually comparable with norm(A) in magnitude, and
 *          BETA always less than and usually comparable with norm(B).
 *
 *  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N)
 *          If JOBVL = 'V', the left eigenvectors u(j) are stored one
 *          after another in the columns of VL, in the same order as
 *          their eigenvalues. If the j-th eigenvalue is real, then
 *          u(j) = VL(:,j), the j-th column of VL. If the j-th and
 *          (j+1)-th eigenvalues form a complex conjugate pair, then
 *          u(j) = VL(:,j)+i*VL(:,j+1) and u(j+1) = VL(:,j)-i*VL(:,j+1).
 *          Each eigenvector is scaled so the largest component has
 *          abs(real part)+abs(imag. part)=1.
 *          Not referenced if JOBVL = 'N'.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the matrix VL. LDVL >= 1, and
 *          if JOBVL = 'V', LDVL >= N.
 *
 *  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N)
 *          If JOBVR = 'V', the right eigenvectors v(j) are stored one
 *          after another in the columns of VR, in the same order as
 *          their eigenvalues. If the j-th eigenvalue is real, then
 *          v(j) = VR(:,j), the j-th column of VR. If the j-th and
 *          (j+1)-th eigenvalues form a complex conjugate pair, then
 *          v(j) = VR(:,j)+i*VR(:,j+1) and v(j+1) = VR(:,j)-i*VR(:,j+1).
 *          Each eigenvector is scaled so the largest component has
 *          abs(real part)+abs(imag. part)=1.
 *          Not referenced if JOBVR = 'N'.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the matrix VR. LDVR >= 1, and
 *          if JOBVR = 'V', LDVR >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,8*N).
 *          For good performance, LWORK must generally be larger.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1,...,N:
 *                The QZ iteration failed.  No eigenvectors have been
 *                calculated, but ALPHAR(j), ALPHAI(j), and BETA(j)
 *                should be correct for j=INFO+1,...,N.
 *          > N:  =N+1: other than QZ iteration failed in DHGEQZ.
 *                =N+2: error return from DTGEVC.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGEV(char jobvl, char jobvr, int n, double* a, int lda, double* b, int ldb, double* alphar, double* alphai, double* beta, double* vl, int ldvl, double* vr, int ldvr, double* work, int lwork)
{
    int info;
    ::F_DGGEV(&jobvl, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGEVX computes for a pair of N-by-N real nonsymmetric matrices (A,B)
 *  the generalized eigenvalues, and optionally, the left and/or right
 *  generalized eigenvectors.
 *
 *  Optionally also, it computes a balancing transformation to improve
 *  the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
 *  LSCALE, RSCALE, ABNRM, and BBNRM), reciprocal condition numbers for
 *  the eigenvalues (RCONDE), and reciprocal condition numbers for the
 *  right eigenvectors (RCONDV).
 *
 *  A generalized eigenvalue for a pair of matrices (A,B) is a scalar
 *  lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
 *  singular. It is usually represented as the pair (alpha,beta), as
 *  there is a reasonable interpretation for beta=0, and even for both
 *  being zero.
 *
 *  The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
 *  of (A,B) satisfies
 *
 *                   A * v(j) = lambda(j) * B * v(j) .
 *
 *  The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
 *  of (A,B) satisfies
 *
 *                   u(j)**H * A  = lambda(j) * u(j)**H * B.
 *
 *  where u(j)**H is the conjugate-transpose of u(j).
 *
 *
 *  Arguments
 *  =========
 *
 *  BALANC  (input) CHARACTER*1
 *          Specifies the balance option to be performed.
 *          = 'N':  do not diagonally scale or permute;
 *          = 'P':  permute only;
 *          = 'S':  scale only;
 *          = 'B':  both permute and scale.
 *          Computed reciprocal condition numbers will be for the
 *          matrices after permuting and/or balancing. Permuting does
 *          not change condition numbers (in exact arithmetic), but
 *          balancing does.
 *
 *  JOBVL   (input) CHARACTER*1
 *          = 'N':  do not compute the left generalized eigenvectors;
 *          = 'V':  compute the left generalized eigenvectors.
 *
 *  JOBVR   (input) CHARACTER*1
 *          = 'N':  do not compute the right generalized eigenvectors;
 *          = 'V':  compute the right generalized eigenvectors.
 *
 *  SENSE   (input) CHARACTER*1
 *          Determines which reciprocal condition numbers are computed.
 *          = 'N': none are computed;
 *          = 'E': computed for eigenvalues only;
 *          = 'V': computed for eigenvectors only;
 *          = 'B': computed for eigenvalues and eigenvectors.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A, B, VL, and VR.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the matrix A in the pair (A,B).
 *          On exit, A has been overwritten. If JOBVL='V' or JOBVR='V'
 *          or both, then A contains the first part of the real Schur
 *          form of the "balanced" versions of the input A and B.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the matrix B in the pair (A,B).
 *          On exit, B has been overwritten. If JOBVL='V' or JOBVR='V'
 *          or both, then B contains the second part of the real Schur
 *          form of the "balanced" versions of the input A and B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of B.  LDB >= max(1,N).
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will
 *          be the generalized eigenvalues.  If ALPHAI(j) is zero, then
 *          the j-th eigenvalue is real; if positive, then the j-th and
 *          (j+1)-st eigenvalues are a complex conjugate pair, with
 *          ALPHAI(j+1) negative.
 *
 *          Note: the quotients ALPHAR(j)/BETA(j) and ALPHAI(j)/BETA(j)
 *          may easily over- or underflow, and BETA(j) may even be zero.
 *          Thus, the user should avoid naively computing the ratio
 *          ALPHA/BETA. However, ALPHAR and ALPHAI will be always less
 *          than and usually comparable with norm(A) in magnitude, and
 *          BETA always less than and usually comparable with norm(B).
 *
 *  VL      (output) DOUBLE PRECISION array, dimension (LDVL,N)
 *          If JOBVL = 'V', the left eigenvectors u(j) are stored one
 *          after another in the columns of VL, in the same order as
 *          their eigenvalues. If the j-th eigenvalue is real, then
 *          u(j) = VL(:,j), the j-th column of VL. If the j-th and
 *          (j+1)-th eigenvalues form a complex conjugate pair, then
 *          u(j) = VL(:,j)+i*VL(:,j+1) and u(j+1) = VL(:,j)-i*VL(:,j+1).
 *          Each eigenvector will be scaled so the largest component have
 *          abs(real part) + abs(imag. part) = 1.
 *          Not referenced if JOBVL = 'N'.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the matrix VL. LDVL >= 1, and
 *          if JOBVL = 'V', LDVL >= N.
 *
 *  VR      (output) DOUBLE PRECISION array, dimension (LDVR,N)
 *          If JOBVR = 'V', the right eigenvectors v(j) are stored one
 *          after another in the columns of VR, in the same order as
 *          their eigenvalues. If the j-th eigenvalue is real, then
 *          v(j) = VR(:,j), the j-th column of VR. If the j-th and
 *          (j+1)-th eigenvalues form a complex conjugate pair, then
 *          v(j) = VR(:,j)+i*VR(:,j+1) and v(j+1) = VR(:,j)-i*VR(:,j+1).
 *          Each eigenvector will be scaled so the largest component have
 *          abs(real part) + abs(imag. part) = 1.
 *          Not referenced if JOBVR = 'N'.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the matrix VR. LDVR >= 1, and
 *          if JOBVR = 'V', LDVR >= N.
 *
 *  ILO     (output) INTEGER
 *  IHI     (output) INTEGER
 *          ILO and IHI are integer values such that on exit
 *          A(i,j) = 0 and B(i,j) = 0 if i > j and
 *          j = 1,...,ILO-1 or i = IHI+1,...,N.
 *          If BALANC = 'N' or 'S', ILO = 1 and IHI = N.
 *
 *  LSCALE  (output) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and scaling factors applied
 *          to the left side of A and B.  If PL(j) is the index of the
 *          row interchanged with row j, and DL(j) is the scaling
 *          factor applied to row j, then
 *            LSCALE(j) = PL(j)  for j = 1,...,ILO-1
 *                      = DL(j)  for j = ILO,...,IHI
 *                      = PL(j)  for j = IHI+1,...,N.
 *          The order in which the interchanges are made is N to IHI+1,
 *          then 1 to ILO-1.
 *
 *  RSCALE  (output) DOUBLE PRECISION array, dimension (N)
 *          Details of the permutations and scaling factors applied
 *          to the right side of A and B.  If PR(j) is the index of the
 *          column interchanged with column j, and DR(j) is the scaling
 *          factor applied to column j, then
 *            RSCALE(j) = PR(j)  for j = 1,...,ILO-1
 *                      = DR(j)  for j = ILO,...,IHI
 *                      = PR(j)  for j = IHI+1,...,N
 *          The order in which the interchanges are made is N to IHI+1,
 *          then 1 to ILO-1.
 *
 *  ABNRM   (output) DOUBLE PRECISION
 *          The one-norm of the balanced matrix A.
 *
 *  BBNRM   (output) DOUBLE PRECISION
 *          The one-norm of the balanced matrix B.
 *
 *  RCONDE  (output) DOUBLE PRECISION array, dimension (N)
 *          If SENSE = 'E' or 'B', the reciprocal condition numbers of
 *          the eigenvalues, stored in consecutive elements of the array.
 *          For a complex conjugate pair of eigenvalues two consecutive
 *          elements of RCONDE are set to the same value. Thus RCONDE(j),
 *          RCONDV(j), and the j-th columns of VL and VR all correspond
 *          to the j-th eigenpair.
 *          If SENSE = 'N or 'V', RCONDE is not referenced.
 *
 *  RCONDV  (output) DOUBLE PRECISION array, dimension (N)
 *          If SENSE = 'V' or 'B', the estimated reciprocal condition
 *          numbers of the eigenvectors, stored in consecutive elements
 *          of the array. For a complex eigenvector two consecutive
 *          elements of RCONDV are set to the same value. If the
 *          eigenvalues cannot be reordered to compute RCONDV(j),
 *          RCONDV(j) is set to 0; this can only occur when the true
 *          value would be very small anyway.
 *          If SENSE = 'N' or 'E', RCONDV is not referenced.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,2*N).
 *          If BALANC = 'S' or 'B', or JOBVL = 'V', or JOBVR = 'V',
 *          LWORK >= max(1,6*N).
 *          If SENSE = 'E' or 'B', LWORK >= max(1,10*N).
 *          If SENSE = 'V' or 'B', LWORK >= 2*N*N+8*N+16.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (N+6)
 *          If SENSE = 'E', IWORK is not referenced.
 *
 *  BWORK   (workspace) LOGICAL array, dimension (N)
 *          If SENSE = 'N', BWORK is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1,...,N:
 *                The QZ iteration failed.  No eigenvectors have been
 *                calculated, but ALPHAR(j), ALPHAI(j), and BETA(j)
 *                should be correct for j=INFO+1,...,N.
 *          > N:  =N+1: other than QZ iteration failed in DHGEQZ.
 *                =N+2: error return from DTGEVC.
 *
 *  Further Details
 *  ===============
 *
 *  Balancing a matrix pair (A,B) includes, first, permuting rows and
 *  columns to isolate eigenvalues, second, applying diagonal similarity
 *  transformation to the rows and columns to make the rows and columns
 *  as close in norm as possible. The computed reciprocal condition
 *  numbers correspond to the balanced matrix. Permuting rows and columns
 *  will not change the condition numbers (in exact arithmetic) but
 *  diagonal scaling will.  For further explanation of balancing, see
 *  section 4.11.1.2 of LAPACK Users' Guide.
 *
 *  An approximate error bound on the chordal distance between the i-th
 *  computed generalized eigenvalue w and the corresponding exact
 *  eigenvalue lambda is
 *
 *       chord(w, lambda) <= EPS * norm(ABNRM, BBNRM) / RCONDE(I)
 *
 *  An approximate error bound for the angle between the i-th computed
 *  eigenvector VL(i) or VR(i) is given by
 *
 *       EPS * norm(ABNRM, BBNRM) / DIF(i).
 *
 *  For further explanation of the reciprocal condition numbers RCONDE
 *  and RCONDV, see section 4.11 of LAPACK User's Guide.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGEVX(char balanc, char jobvl, char jobvr, char sense, int n, double* a, int lda, double* b, int ldb, double* alphar, double* alphai, double* beta, double* vl, int ldvl, double* vr, int ldvr, int* ilo, int* ihi, double* lscale, double* rscale, double* abnrm, double* bbnrm, double* rconde, double* rcondv, double* work, int lwork, int* iwork)
{
    int info;
    ::F_DGGEVX(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, &lwork, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGGLM solves a general Gauss-Markov linear model (GLM) problem:
 *
 *          minimize || y ||_2   subject to   d = A*x + B*y
 *              x
 *
 *  where A is an N-by-M matrix, B is an N-by-P matrix, and d is a
 *  given N-vector. It is assumed that M <= N <= M+P, and
 *
 *             rank(A) = M    and    rank( A B ) = N.
 *
 *  Under these assumptions, the constrained equation is always
 *  consistent, and there is a unique solution x and a minimal 2-norm
 *  solution y, which is obtained using a generalized QR factorization
 *  of the matrices (A, B) given by
 *
 *     A = Q*(R),   B = Q*T*Z.
 *           (0)
 *
 *  In particular, if matrix B is square nonsingular, then the problem
 *  GLM is equivalent to the following weighted linear least squares
 *  problem
 *
 *               minimize || inv(B)*(d-A*x) ||_2
 *                   x
 *
 *  where inv(B) denotes the inverse of B.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The number of rows of the matrices A and B.  N >= 0.
 *
 *  M       (input) INTEGER
 *          The number of columns of the matrix A.  0 <= M <= N.
 *
 *  P       (input) INTEGER
 *          The number of columns of the matrix B.  P >= N-M.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,M)
 *          On entry, the N-by-M matrix A.
 *          On exit, the upper triangular part of the array A contains
 *          the M-by-M upper triangular matrix R.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,P)
 *          On entry, the N-by-P matrix B.
 *          On exit, if N <= P, the upper triangle of the subarray
 *          B(1:N,P-N+1:P) contains the N-by-N upper triangular matrix T;
 *          if N > P, the elements on and above the (N-P)th subdiagonal
 *          contain the N-by-P upper trapezoidal matrix T.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, D is the left hand side of the GLM equation.
 *          On exit, D is destroyed.
 *
 *  X       (output) DOUBLE PRECISION array, dimension (M)
 *  Y       (output) DOUBLE PRECISION array, dimension (P)
 *          On exit, X and Y are the solutions of the GLM problem.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N+M+P).
 *          For optimum performance, LWORK >= M+min(N,P)+max(N,P)*NB,
 *          where NB is an upper bound for the optimal blocksizes for
 *          DGEQRF, SGERQF, DORMQR and SORMRQ.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1:  the upper triangular factor R associated with A in the
 *                generalized QR factorization of the pair (A, B) is
 *                singular, so that rank(A) < M; the least squares
 *                solution could not be computed.
 *          = 2:  the bottom (N-M) by (N-M) part of the upper trapezoidal
 *                factor T associated with B in the generalized QR
 *                factorization of the pair (A, B) is singular, so that
 *                rank( A B ) < N; the least squares solution could not
 *                be computed.
 *
 *  ===================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGGLM(int n, int m, int p, double* a, int lda, double* b, int ldb, double* d, double* x, double* y, double* work, int lwork)
{
    int info;
    ::F_DGGGLM(&n, &m, &p, a, &lda, b, &ldb, d, x, y, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGHRD reduces a pair of real matrices (A,B) to generalized upper
 *  Hessenberg form using orthogonal transformations, where A is a
 *  general matrix and B is upper triangular.  The form of the
 *  generalized eigenvalue problem is
 *     A*x = lambda*B*x,
 *  and B is typically made upper triangular by computing its QR
 *  factorization and moving the orthogonal matrix Q to the left side
 *  of the equation.
 *
 *  This subroutine simultaneously reduces A to a Hessenberg matrix H:
 *     Q**T*A*Z = H
 *  and transforms B to another upper triangular matrix T:
 *     Q**T*B*Z = T
 *  in order to reduce the problem to its standard form
 *     H*y = lambda*T*y
 *  where y = Z**T*x.
 *
 *  The orthogonal matrices Q and Z are determined as products of Givens
 *  rotations.  They may either be formed explicitly, or they may be
 *  postmultiplied into input matrices Q1 and Z1, so that
 *
 *       Q1 * A * Z1**T = (Q1*Q) * H * (Z1*Z)**T
 *
 *       Q1 * B * Z1**T = (Q1*Q) * T * (Z1*Z)**T
 *
 *  If Q1 is the orthogonal matrix from the QR factorization of B in the
 *  original equation A*x = lambda*B*x, then DGGHRD reduces the original
 *  problem to generalized Hessenberg form.
 *
 *  Arguments
 *  =========
 *
 *  COMPQ   (input) CHARACTER*1
 *          = 'N': do not compute Q;
 *          = 'I': Q is initialized to the unit matrix, and the
 *                 orthogonal matrix Q is returned;
 *          = 'V': Q must contain an orthogonal matrix Q1 on entry,
 *                 and the product Q1*Q is returned.
 *
 *  COMPZ   (input) CHARACTER*1
 *          = 'N': do not compute Z;
 *          = 'I': Z is initialized to the unit matrix, and the
 *                 orthogonal matrix Z is returned;
 *          = 'V': Z must contain an orthogonal matrix Z1 on entry,
 *                 and the product Z1*Z is returned.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          ILO and IHI mark the rows and columns of A which are to be
 *          reduced.  It is assumed that A is already upper triangular
 *          in rows and columns 1:ILO-1 and IHI+1:N.  ILO and IHI are
 *          normally set by a previous call to SGGBAL; otherwise they
 *          should be set to 1 and N respectively.
 *          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the N-by-N general matrix to be reduced.
 *          On exit, the upper triangle and the first subdiagonal of A
 *          are overwritten with the upper Hessenberg matrix H, and the
 *          rest is set to zero.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the N-by-N upper triangular matrix B.
 *          On exit, the upper triangular matrix T = Q**T B Z.  The
 *          elements below the diagonal are set to zero.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ, N)
 *          On entry, if COMPQ = 'V', the orthogonal matrix Q1,
 *          typically from the QR factorization of B.
 *          On exit, if COMPQ='I', the orthogonal matrix Q, and if
 *          COMPQ = 'V', the product Q1*Q.
 *          Not referenced if COMPQ='N'.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.
 *          LDQ >= N if COMPQ='V' or 'I'; LDQ >= 1 otherwise.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          On entry, if COMPZ = 'V', the orthogonal matrix Z1.
 *          On exit, if COMPZ='I', the orthogonal matrix Z, and if
 *          COMPZ = 'V', the product Z1*Z.
 *          Not referenced if COMPZ='N'.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.
 *          LDZ >= N if COMPZ='V' or 'I'; LDZ >= 1 otherwise.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  This routine reduces A to Hessenberg and B to triangular form by
 *  an unblocked reduction, as described in _Matrix_Computations_,
 *  by Golub and Van Loan (Johns Hopkins Press.)
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGHRD(char compq, char compz, int n, int ilo, int ihi, double* a, int lda, double* b, int ldb, double* q, int ldq, double* z, int ldz)
{
    int info;
    ::F_DGGHRD(&compq, &compz, &n, &ilo, &ihi, a, &lda, b, &ldb, q, &ldq, z, &ldz, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGLSE solves the linear equality-constrained least squares (LSE)
 *  problem:
 *
 *          minimize || c - A*x ||_2   subject to   B*x = d
 *
 *  where A is an M-by-N matrix, B is a P-by-N matrix, c is a given
 *  M-vector, and d is a given P-vector. It is assumed that
 *  P <= N <= M+P, and
 *
 *           rank(B) = P and  rank( (A) ) = N.
 *                                ( (B) )
 *
 *  These conditions ensure that the LSE problem has a unique solution,
 *  which is obtained using a generalized RQ factorization of the
 *  matrices (B, A) given by
 *
 *     B = (0 R)*Q,   A = Z*T*Q.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrices A and B. N >= 0.
 *
 *  P       (input) INTEGER
 *          The number of rows of the matrix B. 0 <= P <= N <= M+P.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array
 *          contain the min(M,N)-by-N upper trapezoidal matrix T.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the P-by-N matrix B.
 *          On exit, the upper triangle of the subarray B(1:P,N-P+1:N)
 *          contains the P-by-P upper triangular matrix R.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,P).
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (M)
 *          On entry, C contains the right hand side vector for the
 *          least squares part of the LSE problem.
 *          On exit, the residual sum of squares for the solution
 *          is given by the sum of squares of elements N-P+1 to M of
 *          vector C.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (P)
 *          On entry, D contains the right hand side vector for the
 *          constrained equation.
 *          On exit, D is destroyed.
 *
 *  X       (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, X is the solution of the LSE problem.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,M+N+P).
 *          For optimum performance LWORK >= P+min(M,N)+max(M,N)*NB,
 *          where NB is an upper bound for the optimal blocksizes for
 *          DGEQRF, SGERQF, DORMQR and SORMRQ.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1:  the upper triangular factor R associated with B in the
 *                generalized RQ factorization of the pair (B, A) is
 *                singular, so that rank(B) < P; the least squares
 *                solution could not be computed.
 *          = 2:  the (N-P) by (N-P) part of the upper trapezoidal factor
 *                T associated with A in the generalized RQ factorization
 *                of the pair (B, A) is singular, so that
 *                rank( (A) ) < N; the least squares solution could not
 *                    ( (B) )
 *                be computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGLSE(int m, int n, int p, double* a, int lda, double* b, int ldb, double* c, double* d, double* x, double* work, int lwork)
{
    int info;
    ::F_DGGLSE(&m, &n, &p, a, &lda, b, &ldb, c, d, x, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGQRF computes a generalized QR factorization of an N-by-M matrix A
 *  and an N-by-P matrix B:
 *
 *              A = Q*R,        B = Q*T*Z,
 *
 *  where Q is an N-by-N orthogonal matrix, Z is a P-by-P orthogonal
 *  matrix, and R and T assume one of the forms:
 *
 *  if N >= M,  R = ( R11 ) M  ,   or if N < M,  R = ( R11  R12 ) N,
 *                  (  0  ) N-M                         N   M-N
 *                     M
 *
 *  where R11 is upper triangular, and
 *
 *  if N <= P,  T = ( 0  T12 ) N,   or if N > P,  T = ( T11 ) N-P,
 *                   P-N  N                           ( T21 ) P
 *                                                       P
 *
 *  where T12 or T21 is upper triangular.
 *
 *  In particular, if B is square and nonsingular, the GQR factorization
 *  of A and B implicitly gives the QR factorization of inv(B)*A:
 *
 *               inv(B)*A = Z'*(inv(T)*R)
 *
 *  where inv(B) denotes the inverse of the matrix B, and Z' denotes the
 *  transpose of the matrix Z.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The number of rows of the matrices A and B. N >= 0.
 *
 *  M       (input) INTEGER
 *          The number of columns of the matrix A.  M >= 0.
 *
 *  P       (input) INTEGER
 *          The number of columns of the matrix B.  P >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,M)
 *          On entry, the N-by-M matrix A.
 *          On exit, the elements on and above the diagonal of the array
 *          contain the min(N,M)-by-M upper trapezoidal matrix R (R is
 *          upper triangular if N >= M); the elements below the diagonal,
 *          with the array TAUA, represent the orthogonal matrix Q as a
 *          product of min(N,M) elementary reflectors (see Further
 *          Details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  TAUA    (output) DOUBLE PRECISION array, dimension (min(N,M))
 *          The scalar factors of the elementary reflectors which
 *          represent the orthogonal matrix Q (see Further Details).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,P)
 *          On entry, the N-by-P matrix B.
 *          On exit, if N <= P, the upper triangle of the subarray
 *          B(1:N,P-N+1:P) contains the N-by-N upper triangular matrix T;
 *          if N > P, the elements on and above the (N-P)-th subdiagonal
 *          contain the N-by-P upper trapezoidal matrix T; the remaining
 *          elements, with the array TAUB, represent the orthogonal
 *          matrix Z as a product of elementary reflectors (see Further
 *          Details).
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  TAUB    (output) DOUBLE PRECISION array, dimension (min(N,P))
 *          The scalar factors of the elementary reflectors which
 *          represent the orthogonal matrix Z (see Further Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N,M,P).
 *          For optimum performance LWORK >= max(N,M,P)*max(NB1,NB2,NB3),
 *          where NB1 is the optimal blocksize for the QR factorization
 *          of an N-by-M matrix, NB2 is the optimal blocksize for the
 *          RQ factorization of an N-by-P matrix, and NB3 is the optimal
 *          blocksize for a call of DORMQR.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(n,m).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - taua * v * v'
 *
 *  where taua is a real scalar, and v is a real vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
 *  and taua in TAUA(i).
 *  To form Q explicitly, use LAPACK subroutine DORGQR.
 *  To use Q to update another matrix, use LAPACK subroutine DORMQR.
 *
 *  The matrix Z is represented as a product of elementary reflectors
 *
 *     Z = H(1) H(2) . . . H(k), where k = min(n,p).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - taub * v * v'
 *
 *  where taub is a real scalar, and v is a real vector with
 *  v(p-k+i+1:p) = 0 and v(p-k+i) = 1; v(1:p-k+i-1) is stored on exit in
 *  B(n-k+i,1:p-k+i-1), and taub in TAUB(i).
 *  To form Z explicitly, use LAPACK subroutine DORGRQ.
 *  To use Z to update another matrix, use LAPACK subroutine DORMRQ.
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGGQRF(int n, int m, int p, double* a, int lda, double* taua, double* b, int ldb, double* taub, double* work, int lwork)
{
    int info;
    ::F_DGGQRF(&n, &m, &p, a, &lda, taua, b, &ldb, taub, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGRQF computes a generalized RQ factorization of an M-by-N matrix A
 *  and a P-by-N matrix B:
 *
 *              A = R*Q,        B = Z*T*Q,
 *
 *  where Q is an N-by-N orthogonal matrix, Z is a P-by-P orthogonal
 *  matrix, and R and T assume one of the forms:
 *
 *  if M <= N,  R = ( 0  R12 ) M,   or if M > N,  R = ( R11 ) M-N,
 *                   N-M  M                           ( R21 ) N
 *                                                       N
 *
 *  where R12 or R21 is upper triangular, and
 *
 *  if P >= N,  T = ( T11 ) N  ,   or if P < N,  T = ( T11  T12 ) P,
 *                  (  0  ) P-N                         P   N-P
 *                     N
 *
 *  where T11 is upper triangular.
 *
 *  In particular, if B is square and nonsingular, the GRQ factorization
 *  of A and B implicitly gives the RQ factorization of A*inv(B):
 *
 *               A*inv(B) = (R*inv(T))*Z'
 *
 *  where inv(B) denotes the inverse of the matrix B, and Z' denotes the
 *  transpose of the matrix Z.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  P       (input) INTEGER
 *          The number of rows of the matrix B.  P >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrices A and B. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, if M <= N, the upper triangle of the subarray
 *          A(1:M,N-M+1:N) contains the M-by-M upper triangular matrix R;
 *          if M > N, the elements on and above the (M-N)-th subdiagonal
 *          contain the M-by-N upper trapezoidal matrix R; the remaining
 *          elements, with the array TAUA, represent the orthogonal
 *          matrix Q as a product of elementary reflectors (see Further
 *          Details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  TAUA    (output) DOUBLE PRECISION array, dimension (min(M,N))
 *          The scalar factors of the elementary reflectors which
 *          represent the orthogonal matrix Q (see Further Details).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the P-by-N matrix B.
 *          On exit, the elements on and above the diagonal of the array
 *          contain the min(P,N)-by-N upper trapezoidal matrix T (T is
 *          upper triangular if P >= N); the elements below the diagonal,
 *          with the array TAUB, represent the orthogonal matrix Z as a
 *          product of elementary reflectors (see Further Details).
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,P).
 *
 *  TAUB    (output) DOUBLE PRECISION array, dimension (min(P,N))
 *          The scalar factors of the elementary reflectors which
 *          represent the orthogonal matrix Z (see Further Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N,M,P).
 *          For optimum performance LWORK >= max(N,M,P)*max(NB1,NB2,NB3),
 *          where NB1 is the optimal blocksize for the RQ factorization
 *          of an M-by-N matrix, NB2 is the optimal blocksize for the
 *          QR factorization of a P-by-N matrix, and NB3 is the optimal
 *          blocksize for a call of DORMRQ.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INF0= -i, the i-th argument had an illegal value.
 *
 *  Further Details
 *  ===============
 *
 *  The matrix Q is represented as a product of elementary reflectors
 *
 *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - taua * v * v'
 *
 *  where taua is a real scalar, and v is a real vector with
 *  v(n-k+i+1:n) = 0 and v(n-k+i) = 1; v(1:n-k+i-1) is stored on exit in
 *  A(m-k+i,1:n-k+i-1), and taua in TAUA(i).
 *  To form Q explicitly, use LAPACK subroutine DORGRQ.
 *  To use Q to update another matrix, use LAPACK subroutine DORMRQ.
 *
 *  The matrix Z is represented as a product of elementary reflectors
 *
 *     Z = H(1) H(2) . . . H(k), where k = min(p,n).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - taub * v * v'
 *
 *  where taub is a real scalar, and v is a real vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:p) is stored on exit in B(i+1:p,i),
 *  and taub in TAUB(i).
 *  To form Z explicitly, use LAPACK subroutine DORGQR.
 *  To use Z to update another matrix, use LAPACK subroutine DORMQR.
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGGRQF(int m, int p, int n, double* a, int lda, double* taua, double* b, int ldb, double* taub, double* work, int lwork)
{
    int info;
    ::F_DGGRQF(&m, &p, &n, a, &lda, taua, b, &ldb, taub, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGSVD computes the generalized singular value decomposition (GSVD)
 *  of an M-by-N real matrix A and P-by-N real matrix B:
 *
 *      U'*A*Q = D1*( 0 R ),    V'*B*Q = D2*( 0 R )
 *
 *  where U, V and Q are orthogonal matrices, and Z' is the transpose
 *  of Z.  Let K+L = the effective numerical rank of the matrix (A',B')',
 *  then R is a K+L-by-K+L nonsingular upper triangular matrix, D1 and
 *  D2 are M-by-(K+L) and P-by-(K+L) "diagonal" matrices and of the
 *  following structures, respectively:
 *
 *  If M-K-L >= 0,
 *
 *                      K  L
 *         D1 =     K ( I  0 )
 *                  L ( 0  C )
 *              M-K-L ( 0  0 )
 *
 *                    K  L
 *         D2 =   L ( 0  S )
 *              P-L ( 0  0 )
 *
 *                  N-K-L  K    L
 *    ( 0 R ) = K (  0   R11  R12 )
 *              L (  0    0   R22 )
 *
 *  where
 *
 *    C = diag( ALPHA(K+1), ... , ALPHA(K+L) ),
 *    S = diag( BETA(K+1),  ... , BETA(K+L) ),
 *    C**2 + S**2 = I.
 *
 *    R is stored in A(1:K+L,N-K-L+1:N) on exit.
 *
 *  If M-K-L < 0,
 *
 *                    K M-K K+L-M
 *         D1 =   K ( I  0    0   )
 *              M-K ( 0  C    0   )
 *
 *                      K M-K K+L-M
 *         D2 =   M-K ( 0  S    0  )
 *              K+L-M ( 0  0    I  )
 *                P-L ( 0  0    0  )
 *
 *                     N-K-L  K   M-K  K+L-M
 *    ( 0 R ) =     K ( 0    R11  R12  R13  )
 *                M-K ( 0     0   R22  R23  )
 *              K+L-M ( 0     0    0   R33  )
 *
 *  where
 *
 *    C = diag( ALPHA(K+1), ... , ALPHA(M) ),
 *    S = diag( BETA(K+1),  ... , BETA(M) ),
 *    C**2 + S**2 = I.
 *
 *    (R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N), and R33 is stored
 *    ( 0  R22 R23 )
 *    in B(M-K+1:L,N+M-K-L+1:N) on exit.
 *
 *  The routine computes C, S, R, and optionally the orthogonal
 *  transformation matrices U, V and Q.
 *
 *  In particular, if B is an N-by-N nonsingular matrix, then the GSVD of
 *  A and B implicitly gives the SVD of A*inv(B):
 *                       A*inv(B) = U*(D1*inv(D2))*V'.
 *  If ( A',B')' has orthonormal columns, then the GSVD of A and B is
 *  also equal to the CS decomposition of A and B. Furthermore, the GSVD
 *  can be used to derive the solution of the eigenvalue problem:
 *                       A'*A x = lambda* B'*B x.
 *  In some literature, the GSVD of A and B is presented in the form
 *                   U'*A*X = ( 0 D1 ),   V'*B*X = ( 0 D2 )
 *  where U and V are orthogonal and X is nonsingular, D1 and D2 are
 *  ``diagonal''.  The former GSVD form can be converted to the latter
 *  form by taking the nonsingular matrix X as
 *
 *                       X = Q*( I   0    )
 *                             ( 0 inv(R) ).
 *
 *  Arguments
 *  =========
 *
 *  JOBU    (input) CHARACTER*1
 *          = 'U':  Orthogonal matrix U is computed;
 *          = 'N':  U is not computed.
 *
 *  JOBV    (input) CHARACTER*1
 *          = 'V':  Orthogonal matrix V is computed;
 *          = 'N':  V is not computed.
 *
 *  JOBQ    (input) CHARACTER*1
 *          = 'Q':  Orthogonal matrix Q is computed;
 *          = 'N':  Q is not computed.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrices A and B.  N >= 0.
 *
 *  P       (input) INTEGER
 *          The number of rows of the matrix B.  P >= 0.
 *
 *  K       (output) INTEGER
 *  L       (output) INTEGER
 *          On exit, K and L specify the dimension of the subblocks
 *          described in the Purpose section.
 *          K + L = effective numerical rank of (A',B')'.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, A contains the triangular matrix R, or part of R.
 *          See Purpose for details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the P-by-N matrix B.
 *          On exit, B contains the triangular matrix R if M-K-L < 0.
 *          See Purpose for details.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,P).
 *
 *  ALPHA   (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, ALPHA and BETA contain the generalized singular
 *          value pairs of A and B;
 *            ALPHA(1:K) = 1,
 *            BETA(1:K)  = 0,
 *          and if M-K-L >= 0,
 *            ALPHA(K+1:K+L) = C,
 *            BETA(K+1:K+L)  = S,
 *          or if M-K-L < 0,
 *            ALPHA(K+1:M)=C, ALPHA(M+1:K+L)=0
 *            BETA(K+1:M) =S, BETA(M+1:K+L) =1
 *          and
 *            ALPHA(K+L+1:N) = 0
 *            BETA(K+L+1:N)  = 0
 *
 *  U       (output) DOUBLE PRECISION array, dimension (LDU,M)
 *          If JOBU = 'U', U contains the M-by-M orthogonal matrix U.
 *          If JOBU = 'N', U is not referenced.
 *
 *  LDU     (input) INTEGER
 *          The leading dimension of the array U. LDU >= max(1,M) if
 *          JOBU = 'U'; LDU >= 1 otherwise.
 *
 *  V       (output) DOUBLE PRECISION array, dimension (LDV,P)
 *          If JOBV = 'V', V contains the P-by-P orthogonal matrix V.
 *          If JOBV = 'N', V is not referenced.
 *
 *  LDV     (input) INTEGER
 *          The leading dimension of the array V. LDV >= max(1,P) if
 *          JOBV = 'V'; LDV >= 1 otherwise.
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          If JOBQ = 'Q', Q contains the N-by-N orthogonal matrix Q.
 *          If JOBQ = 'N', Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q. LDQ >= max(1,N) if
 *          JOBQ = 'Q'; LDQ >= 1 otherwise.
 *
 *  WORK    (workspace) DOUBLE PRECISION array,
 *                      dimension (max(3*N,M,P)+N)
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (N)
 *          On exit, IWORK stores the sorting information. More
 *          precisely, the following loop will sort ALPHA
 *             for I = K+1, min(M,K+L)
 *                 swap ALPHA(I) and ALPHA(IWORK(I))
 *             endfor
 *          such that ALPHA(1) >= ALPHA(2) >= ... >= ALPHA(N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = 1, the Jacobi-type procedure failed to
 *                converge.  For further details, see subroutine DTGSJA.
 *
 *  Internal Parameters
 *  ===================
 *
 *  TOLA    DOUBLE PRECISION
 *  TOLB    DOUBLE PRECISION
 *          TOLA and TOLB are the thresholds to determine the effective
 *          rank of (A',B')'. Generally, they are set to
 *                   TOLA = MAX(M,N)*norm(A)*MAZHEPS,
 *                   TOLB = MAX(P,N)*norm(B)*MAZHEPS.
 *          The size of TOLA and TOLB may affect the size of backward
 *          errors of the decomposition.
 *
 *  Further Details
 *  ===============
 *
 *  2-96 Based on modifications by
 *     Ming Gu and Huan Ren, Computer Science Division, University of
 *     California at Berkeley, USA
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGGSVD(char jobu, char jobv, char jobq, int m, int n, int p, int* k, int* l, double* a, int lda, double* b, int ldb, double* alpha, double* beta, double* u, int ldu, double* v, int ldv, double* q, int ldq, double* work, int* iwork)
{
    int info;
    ::F_DGGSVD(&jobu, &jobv, &jobq, &m, &n, &p, k, l, a, &lda, b, &ldb, alpha, beta, u, &ldu, v, &ldv, q, &ldq, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGGSVP computes orthogonal matrices U, V and Q such that
 *
 *                   N-K-L  K    L
 *   U'*A*Q =     K ( 0    A12  A13 )  if M-K-L >= 0;
 *                L ( 0     0   A23 )
 *            M-K-L ( 0     0    0  )
 *
 *                   N-K-L  K    L
 *          =     K ( 0    A12  A13 )  if M-K-L < 0;
 *              M-K ( 0     0   A23 )
 *
 *                 N-K-L  K    L
 *   V'*B*Q =   L ( 0     0   B13 )
 *            P-L ( 0     0    0  )
 *
 *  where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
 *  upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0,
 *  otherwise A23 is (M-K)-by-L upper trapezoidal.  K+L = the effective
 *  numerical rank of the (M+P)-by-N matrix (A',B')'.  Z' denotes the
 *  transpose of Z.
 *
 *  This decomposition is the preprocessing step for computing the
 *  Generalized Singular Value Decomposition (GSVD), see subroutine
 *  DGGSVD.
 *
 *  Arguments
 *  =========
 *
 *  JOBU    (input) CHARACTER*1
 *          = 'U':  Orthogonal matrix U is computed;
 *          = 'N':  U is not computed.
 *
 *  JOBV    (input) CHARACTER*1
 *          = 'V':  Orthogonal matrix V is computed;
 *          = 'N':  V is not computed.
 *
 *  JOBQ    (input) CHARACTER*1
 *          = 'Q':  Orthogonal matrix Q is computed;
 *          = 'N':  Q is not computed.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  P       (input) INTEGER
 *          The number of rows of the matrix B.  P >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrices A and B.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, A contains the triangular (or trapezoidal) matrix
 *          described in the Purpose section.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the P-by-N matrix B.
 *          On exit, B contains the triangular matrix described in
 *          the Purpose section.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,P).
 *
 *  TOLA    (input) DOUBLE PRECISION
 *  TOLB    (input) DOUBLE PRECISION
 *          TOLA and TOLB are the thresholds to determine the effective
 *          numerical rank of matrix B and a subblock of A. Generally,
 *          they are set to
 *             TOLA = MAX(M,N)*norm(A)*MAZHEPS,
 *             TOLB = MAX(P,N)*norm(B)*MAZHEPS.
 *          The size of TOLA and TOLB may affect the size of backward
 *          errors of the decomposition.
 *
 *  K       (output) INTEGER
 *  L       (output) INTEGER
 *          On exit, K and L specify the dimension of the subblocks
 *          described in Purpose.
 *          K + L = effective numerical rank of (A',B')'.
 *
 *  U       (output) DOUBLE PRECISION array, dimension (LDU,M)
 *          If JOBU = 'U', U contains the orthogonal matrix U.
 *          If JOBU = 'N', U is not referenced.
 *
 *  LDU     (input) INTEGER
 *          The leading dimension of the array U. LDU >= max(1,M) if
 *          JOBU = 'U'; LDU >= 1 otherwise.
 *
 *  V       (output) DOUBLE PRECISION array, dimension (LDV,P)
 *          If JOBV = 'V', V contains the orthogonal matrix V.
 *          If JOBV = 'N', V is not referenced.
 *
 *  LDV     (input) INTEGER
 *          The leading dimension of the array V. LDV >= max(1,P) if
 *          JOBV = 'V'; LDV >= 1 otherwise.
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          If JOBQ = 'Q', Q contains the orthogonal matrix Q.
 *          If JOBQ = 'N', Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q. LDQ >= max(1,N) if
 *          JOBQ = 'Q'; LDQ >= 1 otherwise.
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  TAU     (workspace) DOUBLE PRECISION array, dimension (N)
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (max(3*N,M,P))
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *
 *  Further Details
 *  ===============
 *
 *  The subroutine uses LAPACK subroutine DGEQPF for the QR factorization
 *  with column pivoting to detect the effective numerical rank of the
 *  a matrix. It may be replaced by a better rank determination strategy.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGGSVP(char jobu, char jobv, char jobq, int m, int p, int n, double* a, int lda, double* b, int ldb, double tola, double tolb, int* k, int* l, double* u, int ldu, double* v, int ldv, double* q, int ldq, int* iwork, double* tau, double* work)
{
    int info;
    ::F_DGGSVP(&jobu, &jobv, &jobq, &m, &p, &n, a, &lda, b, &ldb, &tola, &tolb, k, l, u, &ldu, v, &ldv, q, &ldq, iwork, tau, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGTCON estimates the reciprocal of the condition number of a real
 *  tridiagonal matrix A using the LU factorization as computed by
 *  DGTTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  NORM    (input) CHARACTER*1
 *          Specifies whether the 1-norm condition number or the
 *          infinity-norm condition number is required:
 *          = '1' or 'O':  1-norm;
 *          = 'I':         Infinity-norm.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  DL      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) multipliers that define the matrix L from the
 *          LU factorization of A as computed by DGTTRF.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the upper triangular matrix U from
 *          the LU factorization of A.
 *
 *  DU      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) elements of the first superdiagonal of U.
 *
 *  DU2     (input) DOUBLE PRECISION array, dimension (N-2)
 *          The (n-2) elements of the second superdiagonal of U.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices; for 1 <= i <= n, row i of the matrix was
 *          interchanged with row IPIV(i).  IPIV(i) will always be either
 *          i or i+1; IPIV(i) = i indicates a row interchange was not
 *          required.
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          If NORM = '1' or 'O', the 1-norm of the original matrix A.
 *          If NORM = 'I', the infinity-norm of the original matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGTCON(char norm, int n, double* dl, double* d, double* du, double* du2, int* ipiv, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DGTCON(&norm, &n, dl, d, du, du2, ipiv, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGTRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is tridiagonal, and provides
 *  error bounds and backward error estimates for the solution.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B     (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  DL      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of A.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The diagonal elements of A.
 *
 *  DU      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) superdiagonal elements of A.
 *
 *  DLF     (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) multipliers that define the matrix L from the
 *          LU factorization of A as computed by DGTTRF.
 *
 *  DF      (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the upper triangular matrix U from
 *          the LU factorization of A.
 *
 *  DUF     (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) elements of the first superdiagonal of U.
 *
 *  DU2     (input) DOUBLE PRECISION array, dimension (N-2)
 *          The (n-2) elements of the second superdiagonal of U.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices; for 1 <= i <= n, row i of the matrix was
 *          interchanged with row IPIV(i).  IPIV(i) will always be either
 *          i or i+1; IPIV(i) = i indicates a row interchange was not
 *          required.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DGTTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGTRFS(char trans, int n, int nrhs, double* dl, double* d, double* du, double* dlf, double* df, double* duf, double* du2, int* ipiv, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DGTRFS(&trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGTSV  solves the equation
 *
 *     A*X = B,
 *
 *  where A is an n by n tridiagonal matrix, by Gaussian elimination with
 *  partial pivoting.
 *
 *  Note that the equation  A'*X = B  may be solved by interchanging the
 *  order of the arguments DU and DL.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  DL      (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, DL must contain the (n-1) sub-diagonal elements of
 *          A.
 *
 *          On exit, DL is overwritten by the (n-2) elements of the
 *          second super-diagonal of the upper triangular matrix U from
 *          the LU factorization of A, in DL(1), ..., DL(n-2).
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, D must contain the diagonal elements of A.
 *
 *          On exit, D is overwritten by the n diagonal elements of U.
 *
 *  DU      (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, DU must contain the (n-1) super-diagonal elements
 *          of A.
 *
 *          On exit, DU is overwritten by the (n-1) elements of the first
 *          super-diagonal of U.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N by NRHS matrix of right hand side matrix B.
 *          On exit, if INFO = 0, the N by NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, U(i,i) is exactly zero, and the solution
 *               has not been computed.  The factorization has not been
 *               completed unless i = N.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGTSV(int n, int nrhs, double* dl, double* d, double* du, double* b, int ldb)
{
    int info;
    ::F_DGTSV(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGTSVX uses the LU factorization to compute the solution to a real
 *  system of linear equations A * X = B or A**T * X = B,
 *  where A is a tridiagonal matrix of order N and X and B are N-by-NRHS
 *  matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'N', the LU decomposition is used to factor the matrix A
 *     as A = L * U, where L is a product of permutation and unit lower
 *     bidiagonal matrices and U is upper triangular with nonzeros in
 *     only the main diagonal and first two superdiagonals.
 *
 *  2. If some U(i,i)=0, so that U is exactly singular, then the routine
 *     returns with INFO = i. Otherwise, the factored form of A is used
 *     to estimate the condition number of the matrix A.  If the
 *     reciprocal of the condition number is less than machine precision,
 *  C++ Return value: INFO    (output) INTEGER
 *     to solve for X and compute error bounds as described below.
 *
 *  3. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  4. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of A has been
 *          supplied on entry.
 *          = 'F':  DLF, DF, DUF, DU2, and IPIV contain the factored
 *                  form of A; DL, D, DU, DLF, DF, DUF, DU2 and IPIV
 *                  will not be modified.
 *          = 'N':  The matrix will be copied to DLF, DF, and DUF
 *                  and factored.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B     (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  DL      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of A.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of A.
 *
 *  DU      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) superdiagonal elements of A.
 *
 *  DLF     (input or output) DOUBLE PRECISION array, dimension (N-1)
 *          If FACT = 'F', then DLF is an input argument and on entry
 *          contains the (n-1) multipliers that define the matrix L from
 *          the LU factorization of A as computed by DGTTRF.
 *
 *          If FACT = 'N', then DLF is an output argument and on exit
 *          contains the (n-1) multipliers that define the matrix L from
 *          the LU factorization of A.
 *
 *  DF      (input or output) DOUBLE PRECISION array, dimension (N)
 *          If FACT = 'F', then DF is an input argument and on entry
 *          contains the n diagonal elements of the upper triangular
 *          matrix U from the LU factorization of A.
 *
 *          If FACT = 'N', then DF is an output argument and on exit
 *          contains the n diagonal elements of the upper triangular
 *          matrix U from the LU factorization of A.
 *
 *  DUF     (input or output) DOUBLE PRECISION array, dimension (N-1)
 *          If FACT = 'F', then DUF is an input argument and on entry
 *          contains the (n-1) elements of the first superdiagonal of U.
 *
 *          If FACT = 'N', then DUF is an output argument and on exit
 *          contains the (n-1) elements of the first superdiagonal of U.
 *
 *  DU2     (input or output) DOUBLE PRECISION array, dimension (N-2)
 *          If FACT = 'F', then DU2 is an input argument and on entry
 *          contains the (n-2) elements of the second superdiagonal of
 *          U.
 *
 *          If FACT = 'N', then DU2 is an output argument and on exit
 *          contains the (n-2) elements of the second superdiagonal of
 *          U.
 *
 *  IPIV    (input or output) INTEGER array, dimension (N)
 *          If FACT = 'F', then IPIV is an input argument and on entry
 *          contains the pivot indices from the LU factorization of A as
 *          computed by DGTTRF.
 *
 *          If FACT = 'N', then IPIV is an output argument and on exit
 *          contains the pivot indices from the LU factorization of A;
 *          row i of the matrix was interchanged with row IPIV(i).
 *          IPIV(i) will always be either i or i+1; IPIV(i) = i indicates
 *          a row interchange was not required.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The N-by-NRHS right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A.  If RCOND is less than the machine precision (in
 *          particular, if RCOND = 0), the matrix is singular to working
 *          precision.  This condition is indicated by a return code of
 *  C++ Return value: INFO    (output) INTEGER
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  U(i,i) is exactly zero.  The factorization
 *                       has not been completed unless i = N, but the
 *                       factor U is exactly singular, so the solution
 *                       and error bounds could not be computed.
 *                       RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGTSVX(char fact, char trans, int n, int nrhs, double* dl, double* d, double* du, double* dlf, double* df, double* duf, double* du2, int* ipiv, double* b, int ldb, double* x, int ldx, double* rcond)
{
    int info;
    ::F_DGTSVX(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb, x, &ldx, rcond, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGTTRF computes an LU factorization of a real tridiagonal matrix A
 *  using elimination with partial pivoting and row interchanges.
 *
 *  The factorization has the form
 *     A = L * U
 *  where L is a product of permutation and unit lower bidiagonal
 *  matrices and U is upper triangular with nonzeros in only the main
 *  diagonal and first two superdiagonals.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.
 *
 *  DL      (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, DL must contain the (n-1) sub-diagonal elements of
 *          A.
 *
 *          On exit, DL is overwritten by the (n-1) multipliers that
 *          define the matrix L from the LU factorization of A.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, D must contain the diagonal elements of A.
 *
 *          On exit, D is overwritten by the n diagonal elements of the
 *          upper triangular matrix U from the LU factorization of A.
 *
 *  DU      (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, DU must contain the (n-1) super-diagonal elements
 *          of A.
 *
 *          On exit, DU is overwritten by the (n-1) elements of the first
 *          super-diagonal of U.
 *
 *  DU2     (output) DOUBLE PRECISION array, dimension (N-2)
 *          On exit, DU2 is overwritten by the (n-2) elements of the
 *          second super-diagonal of U.
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          The pivot indices; for 1 <= i <= n, row i of the matrix was
 *          interchanged with row IPIV(i).  IPIV(i) will always be either
 *          i or i+1; IPIV(i) = i indicates a row interchange was not
 *          required.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -k, the k-th argument had an illegal value
 *          > 0:  if INFO = k, U(k,k) is exactly zero. The factorization
 *                has been completed, but the factor U is exactly
 *                singular, and division by zero will occur if it is used
 *                to solve a system of equations.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DGTTRF(int n, double* dl, double* d, double* du, double* du2, int* ipiv)
{
    int info;
    ::F_DGTTRF(&n, dl, d, du, du2, ipiv, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DGTTRS solves one of the systems of equations
 *     A*X = B  or  A'*X = B,
 *  with a tridiagonal matrix A using the LU factorization computed
 *  by DGTTRF.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations.
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A'* X = B  (Transpose)
 *          = 'C':  A'* X = B  (Conjugate transpose = Transpose)
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  DL      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) multipliers that define the matrix L from the
 *          LU factorization of A.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the upper triangular matrix U from
 *          the LU factorization of A.
 *
 *  DU      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) elements of the first super-diagonal of U.
 *
 *  DU2     (input) DOUBLE PRECISION array, dimension (N-2)
 *          The (n-2) elements of the second super-diagonal of U.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          The pivot indices; for 1 <= i <= n, row i of the matrix was
 *          interchanged with row IPIV(i).  IPIV(i) will always be either
 *          i or i+1; IPIV(i) = i indicates a row interchange was not
 *          required.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the matrix of right hand side vectors B.
 *          On exit, B is overwritten by the solution vectors X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DGTTRS(char trans, int n, int nrhs, double* dl, double* d, double* du, double* du2, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DGTTRS(&trans, &n, &nrhs, dl, d, du, du2, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DHGEQZ computes the eigenvalues of a real matrix pair (H,T),
 *  where H is an upper Hessenberg matrix and T is upper triangular,
 *  using the double-shift QZ method.
 *  Matrix pairs of this type are produced by the reduction to
 *  generalized upper Hessenberg form of a real matrix pair (A,B):
 *
 *     A = Q1*H*Z1**T,  B = Q1*T*Z1**T,
 *
 *  as computed by DGGHRD.
 *
 *  If JOB='S', then the Hessenberg-triangular pair (H,T) is
 *  also reduced to generalized Schur form,
 *
 *     H = Q*S*Z**T,  T = Q*P*Z**T,
 *
 *  where Q and Z are orthogonal matrices, P is an upper triangular
 *  matrix, and S is a quasi-triangular matrix with 1-by-1 and 2-by-2
 *  diagonal blocks.
 *
 *  The 1-by-1 blocks correspond to real eigenvalues of the matrix pair
 *  (H,T) and the 2-by-2 blocks correspond to complex conjugate pairs of
 *  eigenvalues.
 *
 *  Additionally, the 2-by-2 upper triangular diagonal blocks of P
 *  corresponding to 2-by-2 blocks of S are reduced to positive diagonal
 *  form, i.e., if S(j+1,j) is non-zero, then P(j+1,j) = P(j,j+1) = 0,
 *  P(j,j) > 0, and P(j+1,j+1) > 0.
 *
 *  Optionally, the orthogonal matrix Q from the generalized Schur
 *  factorization may be postmultiplied into an input matrix Q1, and the
 *  orthogonal matrix Z may be postmultiplied into an input matrix Z1.
 *  If Q1 and Z1 are the orthogonal matrices from DGGHRD that reduced
 *  the matrix pair (A,B) to generalized upper Hessenberg form, then the
 *  output matrices Q1*Q and Z1*Z are the orthogonal factors from the
 *  generalized Schur factorization of (A,B):
 *
 *     A = (Q1*Q)*S*(Z1*Z)**T,  B = (Q1*Q)*P*(Z1*Z)**T.
 *
 *  To avoid overflow, eigenvalues of the matrix pair (H,T) (equivalently,
 *  of (A,B)) are computed as a pair of values (alpha,beta), where alpha is
 *  complex and beta real.
 *  If beta is nonzero, lambda = alpha / beta is an eigenvalue of the
 *  generalized nonsymmetric eigenvalue problem (GNEP)
 *     A*x = lambda*B*x
 *  and if alpha is nonzero, mu = beta / alpha is an eigenvalue of the
 *  alternate form of the GNEP
 *     mu*A*y = B*y.
 *  Real eigenvalues can be read directly from the generalized Schur
 *  form:
 *    alpha = S(i,i), beta = P(i,i).
 *
 *  Ref: C.B. Moler & G.W. Stewart, "An Algorithm for Generalized Matrix
 *       Eigenvalue Problems", SIAM J. Numer. Anal., 10(1973),
 *       pp. 241--256.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          = 'E': Compute eigenvalues only;
 *          = 'S': Compute eigenvalues and the Schur form.
 *
 *  COMPQ   (input) CHARACTER*1
 *          = 'N': Left Schur vectors (Q) are not computed;
 *          = 'I': Q is initialized to the unit matrix and the matrix Q
 *                 of left Schur vectors of (H,T) is returned;
 *          = 'V': Q must contain an orthogonal matrix Q1 on entry and
 *                 the product Q1*Q is returned.
 *
 *  COMPZ   (input) CHARACTER*1
 *          = 'N': Right Schur vectors (Z) are not computed;
 *          = 'I': Z is initialized to the unit matrix and the matrix Z
 *                 of right Schur vectors of (H,T) is returned;
 *          = 'V': Z must contain an orthogonal matrix Z1 on entry and
 *                 the product Z1*Z is returned.
 *
 *  N       (input) INTEGER
 *          The order of the matrices H, T, Q, and Z.  N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          ILO and IHI mark the rows and columns of H which are in
 *          Hessenberg form.  It is assumed that A is already upper
 *          triangular in rows and columns 1:ILO-1 and IHI+1:N.
 *          If N > 0, 1 <= ILO <= IHI <= N; if N = 0, ILO=1 and IHI=0.
 *
 *  H       (input/output) DOUBLE PRECISION array, dimension (LDH, N)
 *          On entry, the N-by-N upper Hessenberg matrix H.
 *          On exit, if JOB = 'S', H contains the upper quasi-triangular
 *          matrix S from the generalized Schur factorization;
 *          2-by-2 diagonal blocks (corresponding to complex conjugate
 *          pairs of eigenvalues) are returned in standard form, with
 *          H(i,i) = H(i+1,i+1) and H(i+1,i)*H(i,i+1) < 0.
 *          If JOB = 'E', the diagonal blocks of H match those of S, but
 *          the rest of H is unspecified.
 *
 *  LDH     (input) INTEGER
 *          The leading dimension of the array H.  LDH >= max( 1, N ).
 *
 *  T       (input/output) DOUBLE PRECISION array, dimension (LDT, N)
 *          On entry, the N-by-N upper triangular matrix T.
 *          On exit, if JOB = 'S', T contains the upper triangular
 *          matrix P from the generalized Schur factorization;
 *          2-by-2 diagonal blocks of P corresponding to 2-by-2 blocks of S
 *          are reduced to positive diagonal form, i.e., if H(j+1,j) is
 *          non-zero, then T(j+1,j) = T(j,j+1) = 0, T(j,j) > 0, and
 *          T(j+1,j+1) > 0.
 *          If JOB = 'E', the diagonal blocks of T match those of P, but
 *          the rest of T is unspecified.
 *
 *  LDT     (input) INTEGER
 *          The leading dimension of the array T.  LDT >= max( 1, N ).
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *          The real parts of each scalar alpha defining an eigenvalue
 *          of GNEP.
 *
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *          The imaginary parts of each scalar alpha defining an
 *          eigenvalue of GNEP.
 *          If ALPHAI(j) is zero, then the j-th eigenvalue is real; if
 *          positive, then the j-th and (j+1)-st eigenvalues are a
 *          complex conjugate pair, with ALPHAI(j+1) = -ALPHAI(j).
 *
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          The scalars beta that define the eigenvalues of GNEP.
 *          Together, the quantities alpha = (ALPHAR(j),ALPHAI(j)) and
 *          beta = BETA(j) represent the j-th eigenvalue of the matrix
 *          pair (A,B), in one of the forms lambda = alpha/beta or
 *          mu = beta/alpha.  Since either lambda or mu may overflow,
 *          they should not, in general, be computed.
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ, N)
 *          On entry, if COMPZ = 'V', the orthogonal matrix Q1 used in
 *          the reduction of (A,B) to generalized Hessenberg form.
 *          On exit, if COMPZ = 'I', the orthogonal matrix of left Schur
 *          vectors of (H,T), and if COMPZ = 'V', the orthogonal matrix
 *          of left Schur vectors of (A,B).
 *          Not referenced if COMPZ = 'N'.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.  LDQ >= 1.
 *          If COMPQ='V' or 'I', then LDQ >= N.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          On entry, if COMPZ = 'V', the orthogonal matrix Z1 used in
 *          the reduction of (A,B) to generalized Hessenberg form.
 *          On exit, if COMPZ = 'I', the orthogonal matrix of
 *          right Schur vectors of (H,T), and if COMPZ = 'V', the
 *          orthogonal matrix of right Schur vectors of (A,B).
 *          Not referenced if COMPZ = 'N'.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1.
 *          If COMPZ='V' or 'I', then LDZ >= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO >= 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,N).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          = 1,...,N: the QZ iteration did not converge.  (H,T) is not
 *                     in Schur form, but ALPHAR(i), ALPHAI(i), and
 *                     BETA(i), i=INFO+1,...,N should be correct.
 *          = N+1,...,2*N: the shift calculation failed.  (H,T) is not
 *                     in Schur form, but ALPHAR(i), ALPHAI(i), and
 *                     BETA(i), i=INFO-N+1,...,N should be correct.
 *
 *  Further Details
 *  ===============
 *
 *  Iteration counters:
 *
 *  JITER  -- counts iterations.
 *  IITER  -- counts iterations run since ILAST was last
 *            changed.  This is therefore reset only when a 1-by-1 or
 *            2-by-2 block deflates off the bottom.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 *    $                     SAFETY = 1.0E+0 )
 **/
int C_DHGEQZ(char job, char compq, char compz, int n, int ilo, int ihi, double* h, int ldh, double* t, int ldt, double* alphar, double* alphai, double* beta, double* q, int ldq, double* z, int ldz, double* work, int lwork)
{
    int info;
    ::F_DHGEQZ(&job, &compq, &compz, &n, &ilo, &ihi, h, &ldh, t, &ldt, alphar, alphai, beta, q, &ldq, z, &ldz, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DHSEIN uses inverse iteration to find specified right and/or left
 *  eigenvectors of a real upper Hessenberg matrix H.
 *
 *  The right eigenvector x and the left eigenvector y of the matrix H
 *  corresponding to an eigenvalue w are defined by:
 *
 *               H * x = w * x,     y**h * H = w * y**h
 *
 *  where y**h denotes the conjugate transpose of the vector y.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'R': compute right eigenvectors only;
 *          = 'L': compute left eigenvectors only;
 *          = 'B': compute both right and left eigenvectors.
 *
 *  EIGSRC  (input) CHARACTER*1
 *          Specifies the source of eigenvalues supplied in (WR,WI):
 *          = 'Q': the eigenvalues were found using DHSEQR; thus, if
 *                 H has zero subdiagonal elements, and so is
 *                 block-triangular, then the j-th eigenvalue can be
 *                 assumed to be an eigenvalue of the block containing
 *                 the j-th row/column.  This property allows DHSEIN to
 *                 perform inverse iteration on just one diagonal block.
 *          = 'N': no assumptions are made on the correspondence
 *                 between eigenvalues and diagonal blocks.  In this
 *                 case, DHSEIN must always perform inverse iteration
 *                 using the whole matrix H.
 *
 *  INITV   (input) CHARACTER*1
 *          = 'N': no initial vectors are supplied;
 *          = 'U': user-supplied initial vectors are stored in the arrays
 *                 VL and/or VR.
 *
 *  SELECT  (input/output) LOGICAL array, dimension (N)
 *          Specifies the eigenvectors to be computed. To select the
 *          real eigenvector corresponding to a real eigenvalue WR(j),
 *          SELECT(j) must be set to .TRUE.. To select the complex
 *          eigenvector corresponding to a complex eigenvalue
 *          (WR(j),WI(j)), with complex conjugate (WR(j+1),WI(j+1)),
 *          either SELECT(j) or SELECT(j+1) or both must be set to
 *          .TRUE.; then on exit SELECT(j) is .TRUE. and SELECT(j+1) is
 *          .FALSE..
 *
 *  N       (input) INTEGER
 *          The order of the matrix H.  N >= 0.
 *
 *  H       (input) DOUBLE PRECISION array, dimension (LDH,N)
 *          The upper Hessenberg matrix H.
 *
 *  LDH     (input) INTEGER
 *          The leading dimension of the array H.  LDH >= max(1,N).
 *
 *  WR      (input/output) DOUBLE PRECISION array, dimension (N)
 *  WI      (input) DOUBLE PRECISION array, dimension (N)
 *          On entry, the real and imaginary parts of the eigenvalues of
 *          H; a complex conjugate pair of eigenvalues must be stored in
 *          consecutive elements of WR and WI.
 *          On exit, WR may have been altered since close eigenvalues
 *          are perturbed slightly in searching for independent
 *          eigenvectors.
 *
 *  VL      (input/output) DOUBLE PRECISION array, dimension (LDVL,MM)
 *          On entry, if INITV = 'U' and SIDE = 'L' or 'B', VL must
 *          contain starting vectors for the inverse iteration for the
 *          left eigenvectors; the starting vector for each eigenvector
 *          must be in the same column(s) in which the eigenvector will
 *          be stored.
 *          On exit, if SIDE = 'L' or 'B', the left eigenvectors
 *          specified by SELECT will be stored consecutively in the
 *          columns of VL, in the same order as their eigenvalues. A
 *          complex eigenvector corresponding to a complex eigenvalue is
 *          stored in two consecutive columns, the first holding the real
 *          part and the second the imaginary part.
 *          If SIDE = 'R', VL is not referenced.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the array VL.
 *          LDVL >= max(1,N) if SIDE = 'L' or 'B'; LDVL >= 1 otherwise.
 *
 *  VR      (input/output) DOUBLE PRECISION array, dimension (LDVR,MM)
 *          On entry, if INITV = 'U' and SIDE = 'R' or 'B', VR must
 *          contain starting vectors for the inverse iteration for the
 *          right eigenvectors; the starting vector for each eigenvector
 *          must be in the same column(s) in which the eigenvector will
 *          be stored.
 *          On exit, if SIDE = 'R' or 'B', the right eigenvectors
 *          specified by SELECT will be stored consecutively in the
 *          columns of VR, in the same order as their eigenvalues. A
 *          complex eigenvector corresponding to a complex eigenvalue is
 *          stored in two consecutive columns, the first holding the real
 *          part and the second the imaginary part.
 *          If SIDE = 'L', VR is not referenced.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR.
 *          LDVR >= max(1,N) if SIDE = 'R' or 'B'; LDVR >= 1 otherwise.
 *
 *  MM      (input) INTEGER
 *          The number of columns in the arrays VL and/or VR. MM >= M.
 *
 *  M       (output) INTEGER
 *          The number of columns in the arrays VL and/or VR required to
 *          store the eigenvectors; each selected real eigenvector
 *          occupies one column and each selected complex eigenvector
 *          occupies two columns.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension ((N+2)*N)
 *
 *  IFAILL  (output) INTEGER array, dimension (MM)
 *          If SIDE = 'L' or 'B', IFAILL(i) = j > 0 if the left
 *          eigenvector in the i-th column of VL (corresponding to the
 *          eigenvalue w(j)) failed to converge; IFAILL(i) = 0 if the
 *          eigenvector converged satisfactorily. If the i-th and (i+1)th
 *          columns of VL hold a complex eigenvector, then IFAILL(i) and
 *          IFAILL(i+1) are set to the same value.
 *          If SIDE = 'R', IFAILL is not referenced.
 *
 *  IFAILR  (output) INTEGER array, dimension (MM)
 *          If SIDE = 'R' or 'B', IFAILR(i) = j > 0 if the right
 *          eigenvector in the i-th column of VR (corresponding to the
 *          eigenvalue w(j)) failed to converge; IFAILR(i) = 0 if the
 *          eigenvector converged satisfactorily. If the i-th and (i+1)th
 *          columns of VR hold a complex eigenvector, then IFAILR(i) and
 *          IFAILR(i+1) are set to the same value.
 *          If SIDE = 'L', IFAILR is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, i is the number of eigenvectors which
 *                failed to converge; see IFAILL and IFAILR for further
 *                details.
 *
 *  Further Details
 *  ===============
 *
 *  Each eigenvector is normalized so that the element of largest
 *  magnitude has magnitude 1; here the magnitude of a complex number
 *  (x,y) is taken to be |x|+|y|.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DHSEIN(char side, char eigsrc, char initv, int n, double* h, int ldh, double* wr, double* wi, double* vl, int ldvl, double* vr, int ldvr, int mm, int* m, double* work, int* ifaill, int* ifailr)
{
    int info;
    ::F_DHSEIN(&side, &eigsrc, &initv, &n, h, &ldh, wr, wi, vl, &ldvl, vr, &ldvr, &mm, m, work, ifaill, ifailr, &info);
    return info;
}

/**
 *     Purpose
 *     =======
 *
 *     DHSEQR computes the eigenvalues of a Hessenberg matrix H
 *     and, optionally, the matrices T and Z from the Schur decomposition
 *     H = Z T Z**T, where T is an upper quasi-triangular matrix (the
 *     Schur form), and Z is the orthogonal matrix of Schur vectors.
 *
 *     Optionally Z may be postmultiplied into an input orthogonal
 *     matrix Q so that this routine can give the Schur factorization
 *     of a matrix A which has been reduced to the Hessenberg form H
 *     by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
 *
 *     Arguments
 *     =========
 *
 *     JOB   (input) CHARACTER*1
 *           = 'E':  compute eigenvalues only;
 *           = 'S':  compute eigenvalues and the Schur form T.
 *
 *     COMPZ (input) CHARACTER*1
 *           = 'N':  no Schur vectors are computed;
 *           = 'I':  Z is initialized to the unit matrix and the matrix Z
 *                   of Schur vectors of H is returned;
 *           = 'V':  Z must contain an orthogonal matrix Q on entry, and
 *                   the product Q*Z is returned.
 *
 *     N     (input) INTEGER
 *           The order of the matrix H.  N .GE. 0.
 *
 *     ILO   (input) INTEGER
 *     IHI   (input) INTEGER
 *           It is assumed that H is already upper triangular in rows
 *           and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
 *           set by a previous call to DGEBAL, and then passed to DGEHRD
 *           when the matrix output by DGEBAL is reduced to Hessenberg
 *           form. Otherwise ILO and IHI should be set to 1 and N
 *           respectively.  If N.GT.0, then 1.LE.ILO.LE.IHI.LE.N.
 *           If N = 0, then ILO = 1 and IHI = 0.
 *
 *     H     (input/output) DOUBLE PRECISION array, dimension (LDH,N)
 *           On entry, the upper Hessenberg matrix H.
 *           On exit, if INFO = 0 and JOB = 'S', then H contains the
 *           upper quasi-triangular matrix T from the Schur decomposition
 *           (the Schur form); 2-by-2 diagonal blocks (corresponding to
 *           complex conjugate pairs of eigenvalues) are returned in
 *           standard form, with H(i,i) = H(i+1,i+1) and
 *           H(i+1,i)*H(i,i+1).LT.0. If INFO = 0 and JOB = 'E', the
 *           contents of H are unspecified on exit.  (The output value of
 *           H when INFO.GT.0 is given under the description of INFO
 *           below.)
 *
 *           Unlike earlier versions of DHSEQR, this subroutine may
 *           explicitly H(i,j) = 0 for i.GT.j and j = 1, 2, ... ILO-1
 *           or j = IHI+1, IHI+2, ... N.
 *
 *     LDH   (input) INTEGER
 *           The leading dimension of the array H. LDH .GE. max(1,N).
 *
 *     WR    (output) DOUBLE PRECISION array, dimension (N)
 *     WI    (output) DOUBLE PRECISION array, dimension (N)
 *           The real and imaginary parts, respectively, of the computed
 *           eigenvalues. If two eigenvalues are computed as a complex
 *           conjugate pair, they are stored in consecutive elements of
 *           WR and WI, say the i-th and (i+1)th, with WI(i) .GT. 0 and
 *           WI(i+1) .LT. 0. If JOB = 'S', the eigenvalues are stored in
 *           the same order as on the diagonal of the Schur form returned
 *           in H, with WR(i) = H(i,i) and, if H(i:i+1,i:i+1) is a 2-by-2
 *           diagonal block, WI(i) = sqrt(-H(i+1,i)*H(i,i+1)) and
 *           WI(i+1) = -WI(i).
 *
 *     Z     (input/output) DOUBLE PRECISION array, dimension (LDZ,N)
 *           If COMPZ = 'N', Z is not referenced.
 *           If COMPZ = 'I', on entry Z need not be set and on exit,
 *           if INFO = 0, Z contains the orthogonal matrix Z of the Schur
 *           vectors of H.  If COMPZ = 'V', on entry Z must contain an
 *           N-by-N matrix Q, which is assumed to be equal to the unit
 *           matrix except for the submatrix Z(ILO:IHI,ILO:IHI). On exit,
 *           if INFO = 0, Z contains Q*Z.
 *           Normally Q is the orthogonal matrix generated by DORGHR
 *           after the call to DGEHRD which formed the Hessenberg matrix
 *           H. (The output value of Z when INFO.GT.0 is given under
 *           the description of INFO below.)
 *
 *     LDZ   (input) INTEGER
 *           The leading dimension of the array Z.  if COMPZ = 'I' or
 *           COMPZ = 'V', then LDZ.GE.MAX(1,N).  Otherwize, LDZ.GE.1.
 *
 *     WORK  (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
 *           On exit, if INFO = 0, WORK(1) returns an estimate of
 *           the optimal value for LWORK.
 *
 *     LWORK (input) INTEGER
 *           The dimension of the array WORK.  LWORK .GE. max(1,N)
 *           is sufficient and delivers very good and sometimes
 *           optimal performance.  However, LWORK as large as 11*N
 *           may be required for optimal performance.  A workspace
 *           query is recommended to determine the optimal workspace
 *           size.
 *
 *           If LWORK = -1, then DHSEQR does a workspace query.
 *           In this case, DHSEQR checks the input parameters and
 *           estimates the optimal workspace size for the given
 *           values of N, ILO and IHI.  The estimate is returned
 *           in WORK(1).  No error message related to LWORK is
 *           issued by XERBLA.  Neither H nor Z are accessed.
 *
 *
 *  C++ Return value: INFO    (output) INTEGER
 *             =  0:  successful exit
 *           .LT. 0:  if INFO = -i, the i-th argument had an illegal
 *                    value
 *           .GT. 0:  if INFO = i, DHSEQR failed to compute all of
 *                the eigenvalues.  Elements 1:ilo-1 and i+1:n of WR
 *                and WI contain those eigenvalues which have been
 *                successfully computed.  (Failures are rare.)
 *
 *                If INFO .GT. 0 and JOB = 'E', then on exit, the
 *                remaining unconverged eigenvalues are the eigen-
 *                values of the upper Hessenberg matrix rows and
 *                columns ILO through INFO of the final, output
 *                value of H.
 *
 *                If INFO .GT. 0 and JOB   = 'S', then on exit
 *
 *           (*)  (initial value of H)*U  = U*(final value of H)
 *
 *                where U is an orthogonal matrix.  The final
 *                value of H is upper Hessenberg and quasi-triangular
 *                in rows and columns INFO+1 through IHI.
 *
 *                If INFO .GT. 0 and COMPZ = 'V', then on exit
 *
 *                  (final value of Z)  =  (initial value of Z)*U
 *
 *                where U is the orthogonal matrix in (*) (regard-
 *                less of the value of JOB.)
 *
 *                If INFO .GT. 0 and COMPZ = 'I', then on exit
 *                      (final value of Z)  = U
 *                where U is the orthogonal matrix in (*) (regard-
 *                less of the value of JOB.)
 *
 *                If INFO .GT. 0 and COMPZ = 'N', then Z is not
 *                accessed.
 *
 *     ================================================================
 *             Default values supplied by
 *             ILAENV(ISPEC,'DHSEQR',JOB(:1)//COMPZ(:1),N,ILO,IHI,LWORK).
 *             It is suggested that these defaults be adjusted in order
 *             to attain best performance in each particular
 *             computational environment.
 *
 *            ISPEC=12: The DLAHQR vs DLAQR0 crossover point.
 *                      Default: 75. (Must be at least 11.)
 *
 *            ISPEC=13: Recommended deflation window size.
 *                      This depends on ILO, IHI and NS.  NS is the
 *                      number of simultaneous shifts returned
 *                      by ILAENV(ISPEC=15).  (See ISPEC=15 below.)
 *                      The default for (IHI-ILO+1).LE.500 is NS.
 *                      The default for (IHI-ILO+1).GT.500 is 3*NS/2.
 *
 *            ISPEC=14: Nibble crossover point. (See IPARMQ for
 *                      details.)  Default: 14% of deflation window
 *                      size.
 *
 *            ISPEC=15: Number of simultaneous shifts in a multishift
 *                      QR iteration.
 *
 *                      If IHI-ILO+1 is ...
 *
 *                      greater than      ...but less    ... the
 *                      or equal to ...      than        default is
 *
 *                           1               30          NS =   2(+)
 *                          30               60          NS =   4(+)
 *                          60              150          NS =  10(+)
 *                         150              590          NS =  **
 *                         590             3000          NS =  64
 *                        3000             6000          NS = 128
 *                        6000             infinity      NS = 256
 *
 *                  (+)  By default some or all matrices of this order
 *                       are passed to the implicit double shift routine
 *                       DLAHQR and this parameter is ignored.  See
 *                       ISPEC=12 above and comments in IPARMQ for
 *                       details.
 *
 *                 (**)  The asterisks (**) indicate an ad-hoc
 *                       function of N increasing from 10 to 64.
 *
 *            ISPEC=16: Select structured matrix multiply.
 *                      If the number of simultaneous shifts (specified
 *                      by ISPEC=15) is less than 14, then the default
 *                      for ISPEC=16 is 0.  Otherwise the default for
 *                      ISPEC=16 is 2.
 *
 *     ================================================================
 *     Based on contributions by
 *        Karen Braman and Ralph Byers, Department of Mathematics,
 *        University of Kansas, USA
 *
 *     ================================================================
 *     References:
 *       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
 *       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3
 *       Performance, SIAM Journal of Matrix Analysis, volume 23, pages
 *       929--947, 2002.
 *
 *       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
 *       Algorithm Part II: Aggressive Early Deflation, SIAM Journal
 *       of Matrix Analysis, volume 23, pages 948--973, 2002.
 *
 *     ================================================================
 *     .. Parameters ..
 *
 *     ==== Matrices of order NTINY or smaller must be processed by
 *     .    DLAHQR because of insufficient subdiagonal scratch space.
 *     .    (This is a hard limit.) ====
 **/
int C_DHSEQR(char job, char compz, int n, int ilo, int ihi, double* h, int ldh, double* wr, double* wi, double* z, int ldz, double* work, int lwork)
{
    int info;
    ::F_DHSEQR(&job, &compz, &n, &ilo, &ihi, h, &ldh, wr, wi, z, &ldz, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DOPGTR generates a real orthogonal matrix Q which is defined as the
 *  product of n-1 elementary reflectors H(i) of order n, as returned by
 *  DSPTRD using packed storage:
 *
 *  if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
 *
 *  if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U': Upper triangular packed storage used in previous
 *                 call to DSPTRD;
 *          = 'L': Lower triangular packed storage used in previous
 *                 call to DSPTRD.
 *
 *  N       (input) INTEGER
 *          The order of the matrix Q. N >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The vectors which define the elementary reflectors, as
 *          returned by DSPTRD.
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (N-1)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DSPTRD.
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          The N-by-N orthogonal matrix Q.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q. LDQ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N-1)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DOPGTR(char uplo, int n, double* ap, double* tau, double* q, int ldq, double* work)
{
    int info;
    ::F_DOPGTR(&uplo, &n, ap, tau, q, &ldq, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DOPMTR overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix of order nq, with nq = m if
 *  SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
 *  nq-1 elementary reflectors, as returned by DSPTRD using packed
 *  storage:
 *
 *  if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
 *
 *  if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U': Upper triangular packed storage used in previous
 *                 call to DSPTRD;
 *          = 'L': Lower triangular packed storage used in previous
 *                 call to DSPTRD.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension
 *                               (M*(M+1)/2) if SIDE = 'L'
 *                               (N*(N+1)/2) if SIDE = 'R'
 *          The vectors which define the elementary reflectors, as
 *          returned by DSPTRD.  AP is modified by the routine but
 *          restored on exit.
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (M-1) if SIDE = 'L'
 *                                     or (N-1) if SIDE = 'R'
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DSPTRD.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension
 *                                   (N) if SIDE = 'L'
 *                                   (M) if SIDE = 'R'
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DOPMTR(char side, char uplo, char trans, int m, int n, double* ap, double* tau, double* c, int ldc, double* work)
{
    int info;
    ::F_DOPMTR(&side, &uplo, &trans, &m, &n, ap, tau, c, &ldc, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORGBR generates one of the real orthogonal matrices Q or P**T
 *  determined by DGEBRD when reducing a real matrix A to bidiagonal
 *  form: A = Q * B * P**T.  Q and P**T are defined as products of
 *  elementary reflectors H(i) or G(i) respectively.
 *
 *  If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
 *  is of order M:
 *  if m >= k, Q = H(1) H(2) . . . H(k) and DORGBR returns the first n
 *  columns of Q, where m >= n >= k;
 *  if m < k, Q = H(1) H(2) . . . H(m-1) and DORGBR returns Q as an
 *  M-by-M matrix.
 *
 *  If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**T
 *  is of order N:
 *  if k < n, P**T = G(k) . . . G(2) G(1) and DORGBR returns the first m
 *  rows of P**T, where n >= m >= k;
 *  if k >= n, P**T = G(n-1) . . . G(2) G(1) and DORGBR returns P**T as
 *  an N-by-N matrix.
 *
 *  Arguments
 *  =========
 *
 *  VECT    (input) CHARACTER*1
 *          Specifies whether the matrix Q or the matrix P**T is
 *          required, as defined in the transformation applied by DGEBRD:
 *          = 'Q':  generate Q;
 *          = 'P':  generate P**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix Q or P**T to be returned.
 *          M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix Q or P**T to be returned.
 *          N >= 0.
 *          If VECT = 'Q', M >= N >= min(M,K);
 *          if VECT = 'P', N >= M >= min(N,K).
 *
 *  K       (input) INTEGER
 *          If VECT = 'Q', the number of columns in the original M-by-K
 *          matrix reduced by DGEBRD.
 *          If VECT = 'P', the number of rows in the original K-by-N
 *          matrix reduced by DGEBRD.
 *          K >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the vectors which define the elementary reflectors,
 *          as returned by DGEBRD.
 *          On exit, the M-by-N matrix Q or P**T.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension
 *                                (min(M,K)) if VECT = 'Q'
 *                                (min(N,K)) if VECT = 'P'
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i) or G(i), which determines Q or P**T, as
 *          returned by DGEBRD in its array argument TAUQ or TAUP.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,min(M,N)).
 *          For optimum performance LWORK >= min(M,N)*NB, where NB
 *          is the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORGBR(char vect, int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DORGBR(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORGHR generates a real orthogonal matrix Q which is defined as the
 *  product of IHI-ILO elementary reflectors of order N, as returned by
 *  DGEHRD:
 *
 *  Q = H(ilo) H(ilo+1) . . . H(ihi-1).
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix Q. N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          ILO and IHI must have the same values as in the previous call
 *          of DGEHRD. Q is equal to the unit matrix except in the
 *          submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the vectors which define the elementary reflectors,
 *          as returned by DGEHRD.
 *          On exit, the N-by-N orthogonal matrix Q.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (N-1)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGEHRD.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= IHI-ILO.
 *          For optimum performance LWORK >= (IHI-ILO)*NB, where NB is
 *          the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORGHR(int n, int ilo, int ihi, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DORGHR(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORGLQ generates an M-by-N real matrix Q with orthonormal rows,
 *  which is defined as the first M rows of a product of K elementary
 *  reflectors of order N
 *
 *        Q  =  H(k) . . . H(2) H(1)
 *
 *  as returned by DGELQF.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix Q. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix Q. N >= M.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines the
 *          matrix Q. M >= K >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the i-th row must contain the vector which defines
 *          the elementary reflector H(i), for i = 1,2,...,k, as returned
 *          by DGELQF in the first k rows of its array argument A.
 *          On exit, the M-by-N matrix Q.
 *
 *  LDA     (input) INTEGER
 *          The first dimension of the array A. LDA >= max(1,M).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGELQF.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,M).
 *          For optimum performance LWORK >= M*NB, where NB is
 *          the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument has an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORGLQ(int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DORGLQ(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORGQL generates an M-by-N real matrix Q with orthonormal columns,
 *  which is defined as the last N columns of a product of K elementary
 *  reflectors of order M
 *
 *        Q  =  H(k) . . . H(2) H(1)
 *
 *  as returned by DGEQLF.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix Q. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix Q. M >= N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines the
 *          matrix Q. N >= K >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the (n-k+i)-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by DGEQLF in the last k columns of its array
 *          argument A.
 *          On exit, the M-by-N matrix Q.
 *
 *  LDA     (input) INTEGER
 *          The first dimension of the array A. LDA >= max(1,M).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGEQLF.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N).
 *          For optimum performance LWORK >= N*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument has an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
//int C_DORGQL(int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork)
//{
//    int info;
//    ::F_DORGQL(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
//    return info;
//}

/**
 *  Purpose
 *  =======
 *
 *  DORGQR generates an M-by-N real matrix Q with orthonormal columns,
 *  which is defined as the first N columns of a product of K elementary
 *  reflectors of order M
 *
 *        Q  =  H(1) H(2) . . . H(k)
 *
 *  as returned by DGEQRF.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix Q. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix Q. M >= N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines the
 *          matrix Q. N >= K >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by DGEQRF in the first k columns of its array
 *          argument A.
 *          On exit, the M-by-N matrix Q.
 *
 *  LDA     (input) INTEGER
 *          The first dimension of the array A. LDA >= max(1,M).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGEQRF.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N).
 *          For optimum performance LWORK >= N*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument has an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORGQR(int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DORGQR(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORGRQ generates an M-by-N real matrix Q with orthonormal rows,
 *  which is defined as the last M rows of a product of K elementary
 *  reflectors of order N
 *
 *        Q  =  H(1) H(2) . . . H(k)
 *
 *  as returned by DGERQF.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix Q. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix Q. N >= M.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines the
 *          matrix Q. M >= K >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the (m-k+i)-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by DGERQF in the last k rows of its array argument
 *          A.
 *          On exit, the M-by-N matrix Q.
 *
 *  LDA     (input) INTEGER
 *          The first dimension of the array A. LDA >= max(1,M).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGERQF.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,M).
 *          For optimum performance LWORK >= M*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument has an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORGRQ(int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DORGRQ(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORGTR generates a real orthogonal matrix Q which is defined as the
 *  product of n-1 elementary reflectors of order N, as returned by
 *  DSYTRD:
 *
 *  if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
 *
 *  if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U': Upper triangle of A contains elementary reflectors
 *                 from DSYTRD;
 *          = 'L': Lower triangle of A contains elementary reflectors
 *                 from DSYTRD.
 *
 *  N       (input) INTEGER
 *          The order of the matrix Q. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the vectors which define the elementary reflectors,
 *          as returned by DSYTRD.
 *          On exit, the N-by-N orthogonal matrix Q.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (N-1)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DSYTRD.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N-1).
 *          For optimum performance LWORK >= (N-1)*NB, where NB is
 *          the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORGTR(char uplo, int n, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DORGTR(&uplo, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  If VECT = 'Q', DORMBR overwrites the general real M-by-N matrix C
 *  with
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  If VECT = 'P', DORMBR overwrites the general real M-by-N matrix C
 *  with
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      P * C          C * P
 *  TRANS = 'T':      P**T * C       C * P**T
 *
 *  Here Q and P**T are the orthogonal matrices determined by DGEBRD when
 *  reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
 *  P**T are defined as products of elementary reflectors H(i) and G(i)
 *  respectively.
 *
 *  Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
 *  order of the orthogonal matrix Q or P**T that is applied.
 *
 *  If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
 *  if nq >= k, Q = H(1) H(2) . . . H(k);
 *  if nq < k, Q = H(1) H(2) . . . H(nq-1).
 *
 *  If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
 *  if k < nq, P = G(1) G(2) . . . G(k);
 *  if k >= nq, P = G(1) G(2) . . . G(nq-1).
 *
 *  Arguments
 *  =========
 *
 *  VECT    (input) CHARACTER*1
 *          = 'Q': apply Q or Q**T;
 *          = 'P': apply P or P**T.
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q, Q**T, P or P**T from the Left;
 *          = 'R': apply Q, Q**T, P or P**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q  or P;
 *          = 'T':  Transpose, apply Q**T or P**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          If VECT = 'Q', the number of columns in the original
 *          matrix reduced by DGEBRD.
 *          If VECT = 'P', the number of rows in the original
 *          matrix reduced by DGEBRD.
 *          K >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                                (LDA,min(nq,K)) if VECT = 'Q'
 *                                (LDA,nq)        if VECT = 'P'
 *          The vectors which define the elementary reflectors H(i) and
 *          G(i), whose products determine the matrices Q and P, as
 *          returned by DGEBRD.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.
 *          If VECT = 'Q', LDA >= max(1,nq);
 *          if VECT = 'P', LDA >= max(1,min(nq,K)).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (min(nq,K))
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i) or G(i) which determines Q or P, as returned
 *          by DGEBRD in the array argument TAUQ or TAUP.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q
 *          or P*C or P**T*C or C*P or C*P**T.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DORMBR(char vect, char side, char trans, int m, int n, int k, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMBR(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMHR overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix of order nq, with nq = m if
 *  SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
 *  IHI-ILO elementary reflectors, as returned by DGEHRD:
 *
 *  Q = H(ilo) H(ilo+1) . . . H(ihi-1).
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  ILO     (input) INTEGER
 *  IHI     (input) INTEGER
 *          ILO and IHI must have the same values as in the previous call
 *          of DGEHRD. Q is equal to the unit matrix except in the
 *          submatrix Q(ilo+1:ihi,ilo+1:ihi).
 *          If SIDE = 'L', then 1 <= ILO <= IHI <= M, if M > 0, and
 *          ILO = 1 and IHI = 0, if M = 0;
 *          if SIDE = 'R', then 1 <= ILO <= IHI <= N, if N > 0, and
 *          ILO = 1 and IHI = 0, if N = 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                               (LDA,M) if SIDE = 'L'
 *                               (LDA,N) if SIDE = 'R'
 *          The vectors which define the elementary reflectors, as
 *          returned by DGEHRD.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.
 *          LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'.
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension
 *                               (M-1) if SIDE = 'L'
 *                               (N-1) if SIDE = 'R'
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGEHRD.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DORMHR(char side, char trans, int m, int n, int ilo, int ihi, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMHR(&side, &trans, &m, &n, &ilo, &ihi, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMLQ overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(k) . . . H(2) H(1)
 *
 *  as returned by DGELQF. Q is of order M if SIDE = 'L' and of order N
 *  if SIDE = 'R'.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If SIDE = 'L', M >= K >= 0;
 *          if SIDE = 'R', N >= K >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                               (LDA,M) if SIDE = 'L',
 *                               (LDA,N) if SIDE = 'R'
 *          The i-th row must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          DGELQF in the first k rows of its array argument A.
 *          A is modified by the routine but restored on exit.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,K).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGELQF.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORMLQ(char side, char trans, int m, int n, int k, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMLQ(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMQL overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(k) . . . H(2) H(1)
 *
 *  as returned by DGEQLF. Q is of order M if SIDE = 'L' and of order N
 *  if SIDE = 'R'.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If SIDE = 'L', M >= K >= 0;
 *          if SIDE = 'R', N >= K >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,K)
 *          The i-th column must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          DGEQLF in the last k columns of its array argument A.
 *          A is modified by the routine but restored on exit.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.
 *          If SIDE = 'L', LDA >= max(1,M);
 *          if SIDE = 'R', LDA >= max(1,N).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGEQLF.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORMQL(char side, char trans, int m, int n, int k, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMQL(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMQR overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by DGEQRF. Q is of order M if SIDE = 'L' and of order N
 *  if SIDE = 'R'.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If SIDE = 'L', M >= K >= 0;
 *          if SIDE = 'R', N >= K >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,K)
 *          The i-th column must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          DGEQRF in the first k columns of its array argument A.
 *          A is modified by the routine but restored on exit.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.
 *          If SIDE = 'L', LDA >= max(1,M);
 *          if SIDE = 'R', LDA >= max(1,N).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGEQRF.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORMQR(char side, char trans, int m, int n, int k, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMQR(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMR3 overwrites the general real m by n matrix C with
 *
 *        Q * C  if SIDE = 'L' and TRANS = 'N', or
 *
 *        Q'* C  if SIDE = 'L' and TRANS = 'T', or
 *
 *        C * Q  if SIDE = 'R' and TRANS = 'N', or
 *
 *        C * Q' if SIDE = 'R' and TRANS = 'T',
 *
 *  where Q is a real orthogonal matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by DTZRZF. Q is of order m if SIDE = 'L' and of order n
 *  if SIDE = 'R'.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q' from the Left
 *          = 'R': apply Q or Q' from the Right
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N': apply Q  (No transpose)
 *          = 'T': apply Q' (Transpose)
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If SIDE = 'L', M >= K >= 0;
 *          if SIDE = 'R', N >= K >= 0.
 *
 *  L       (input) INTEGER
 *          The number of columns of the matrix A containing
 *          the meaningful part of the Householder reflectors.
 *          If SIDE = 'L', M >= L >= 0, if SIDE = 'R', N >= L >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                               (LDA,M) if SIDE = 'L',
 *                               (LDA,N) if SIDE = 'R'
 *          The i-th row must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          DTZRZF in the last k rows of its array argument A.
 *          A is modified by the routine but restored on exit.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,K).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DTZRZF.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the m-by-n matrix C.
 *          On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension
 *                                   (N) if SIDE = 'L',
 *                                   (M) if SIDE = 'R'
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DORMR3(char side, char trans, int m, int n, int k, int l, double* a, int lda, double* tau, double* c, int ldc, double* work)
{
    int info;
    ::F_DORMR3(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMRQ overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by DGERQF. Q is of order M if SIDE = 'L' and of order N
 *  if SIDE = 'R'.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If SIDE = 'L', M >= K >= 0;
 *          if SIDE = 'R', N >= K >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                               (LDA,M) if SIDE = 'L',
 *                               (LDA,N) if SIDE = 'R'
 *          The i-th row must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          DGERQF in the last k rows of its array argument A.
 *          A is modified by the routine but restored on exit.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,K).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DGERQF.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORMRQ(char side, char trans, int m, int n, int k, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMRQ(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMRZ overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by DTZRZF. Q is of order M if SIDE = 'L' and of order N
 *  if SIDE = 'R'.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  K       (input) INTEGER
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If SIDE = 'L', M >= K >= 0;
 *          if SIDE = 'R', N >= K >= 0.
 *
 *  L       (input) INTEGER
 *          The number of columns of the matrix A containing
 *          the meaningful part of the Householder reflectors.
 *          If SIDE = 'L', M >= L >= 0, if SIDE = 'R', N >= L >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                               (LDA,M) if SIDE = 'L',
 *                               (LDA,N) if SIDE = 'R'
 *          The i-th row must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          DTZRZF in the last k rows of its array argument A.
 *          A is modified by the routine but restored on exit.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,K).
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension (K)
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DTZRZF.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DORMRZ(char side, char trans, int m, int n, int k, int l, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMRZ(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DORMTR overwrites the general real M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'T':      Q**T * C       C * Q**T
 *
 *  where Q is a real orthogonal matrix of order nq, with nq = m if
 *  SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
 *  nq-1 elementary reflectors, as returned by DSYTRD:
 *
 *  if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
 *
 *  if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'L': apply Q or Q**T from the Left;
 *          = 'R': apply Q or Q**T from the Right.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U': Upper triangle of A contains elementary reflectors
 *                 from DSYTRD;
 *          = 'L': Lower triangle of A contains elementary reflectors
 *                 from DSYTRD.
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N':  No transpose, apply Q;
 *          = 'T':  Transpose, apply Q**T.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix C. N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension
 *                               (LDA,M) if SIDE = 'L'
 *                               (LDA,N) if SIDE = 'R'
 *          The vectors which define the elementary reflectors, as
 *          returned by DSYTRD.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.
 *          LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'.
 *
 *  TAU     (input) DOUBLE PRECISION array, dimension
 *                               (M-1) if SIDE = 'L'
 *                               (N-1) if SIDE = 'R'
 *          TAU(i) must contain the scalar factor of the elementary
 *          reflector H(i), as returned by DSYTRD.
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N matrix C.
 *          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If SIDE = 'L', LWORK >= max(1,N);
 *          if SIDE = 'R', LWORK >= max(1,M).
 *          For optimum performance LWORK >= N*NB if SIDE = 'L', and
 *          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
 *          blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DORMTR(char side, char uplo, char trans, int m, int n, double* a, int lda, double* tau, double* c, int ldc, double* work, int lwork)
{
    int info;
    ::F_DORMTR(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBCON estimates the reciprocal of the condition number (in the
 *  1-norm) of a real symmetric positive definite band matrix using the
 *  Cholesky factorization A = U**T*U or A = L*L**T computed by DPBTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangular factor stored in AB;
 *          = 'L':  Lower triangular factor stored in AB.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T of the band matrix A, stored in the
 *          first KD+1 rows of the array.  The j-th column of U or L is
 *          stored in the j-th column of the array AB as follows:
 *          if UPLO ='U', AB(kd+1+i-j,j) = U(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO ='L', AB(1+i-j,j)    = L(i,j) for j<=i<=min(n,j+kd).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          The 1-norm (or infinity-norm) of the symmetric band matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPBCON(char uplo, int n, int kd, double* ab, int ldab, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DPBCON(&uplo, &n, &kd, ab, &ldab, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBEQU computes row and column scalings intended to equilibrate a
 *  symmetric positive definite band matrix A and reduce its condition
 *  number (with respect to the two-norm).  S contains the scale factors,
 *  S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
 *  elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
 *  choice of S puts the condition number of B within a factor N of the
 *  smallest possible condition number over all possible diagonal
 *  scalings.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangular of A is stored;
 *          = 'L':  Lower triangular of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The upper or lower triangle of the symmetric band matrix A,
 *          stored in the first KD+1 rows of the array.  The j-th column
 *          of A is stored in the j-th column of the array AB as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array A.  LDAB >= KD+1.
 *
 *  S       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, S contains the scale factors for A.
 *
 *  SCOND   (output) DOUBLE PRECISION
 *          If INFO = 0, S contains the ratio of the smallest S(i) to
 *          the largest S(i).  If SCOND >= 0.1 and AMAX is neither too
 *          large nor too small, it is not worth scaling by S.
 *
 *  AMAX    (output) DOUBLE PRECISION
 *          Absolute value of largest matrix element.  If AMAX is very
 *          close to overflow or very close to underflow, the matrix
 *          should be scaled.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, the i-th diagonal element is nonpositive.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPBEQU(char uplo, int n, int kd, double* ab, int ldab, double* s, double* scond, double* amax)
{
    int info;
    ::F_DPBEQU(&uplo, &n, &kd, ab, &ldab, s, scond, amax, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is symmetric positive definite
 *  and banded, and provides error bounds and backward error estimates
 *  for the solution.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The upper or lower triangle of the symmetric band matrix A,
 *          stored in the first KD+1 rows of the array.  The j-th column
 *          of A is stored in the j-th column of the array AB as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  AFB     (input) DOUBLE PRECISION array, dimension (LDAFB,N)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T of the band matrix A as computed by
 *          DPBTRF, in the same storage format as A (see AB).
 *
 *  LDAFB   (input) INTEGER
 *          The leading dimension of the array AFB.  LDAFB >= KD+1.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DPBTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPBRFS(char uplo, int n, int kd, int nrhs, double* ab, int ldab, double* afb, int ldafb, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DPBRFS(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBSTF computes a split Cholesky factorization of a real
 *  symmetric positive definite band matrix A.
 *
 *  This routine is designed to be used in conjunction with DSBGST.
 *
 *  The factorization has the form  A = S**T*S  where S is a band matrix
 *  of the same bandwidth as A and the following structure:
 *
 *    S = ( U    )
 *        ( M  L )
 *
 *  where U is upper triangular of order m = (n+kd)/2, and L is lower
 *  triangular of order n-m.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first kd+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *          On exit, if INFO = 0, the factor S from the split Cholesky
 *          factorization A = S**T*S. See Further Details.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, the factorization could not be completed,
 *               because the updated element a(i,i) was negative; the
 *               matrix A is not positive definite.
 *
 *  Further Details
 *  ===============
 *
 *  The band storage scheme is illustrated by the following example, when
 *  N = 7, KD = 2:
 *
 *  S = ( s11  s12  s13                     )
 *      (      s22  s23  s24                )
 *      (           s33  s34                )
 *      (                s44                )
 *      (           s53  s54  s55           )
 *      (                s64  s65  s66      )
 *      (                     s75  s76  s77 )
 *
 *  If UPLO = 'U', the array AB holds:
 *
 *  on entry:                          on exit:
 *
 *   *    *   a13  a24  a35  a46  a57   *    *   s13  s24  s53  s64  s75
 *   *   a12  a23  a34  a45  a56  a67   *   s12  s23  s34  s54  s65  s76
 *  a11  a22  a33  a44  a55  a66  a77  s11  s22  s33  s44  s55  s66  s77
 *
 *  If UPLO = 'L', the array AB holds:
 *
 *  on entry:                          on exit:
 *
 *  a11  a22  a33  a44  a55  a66  a77  s11  s22  s33  s44  s55  s66  s77
 *  a21  a32  a43  a54  a65  a76   *   s12  s23  s34  s54  s65  s76   *
 *  a31  a42  a53  a64  a64   *    *   s13  s24  s53  s64  s75   *    *
 *
 *  Array elements marked * are not used by the routine.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPBSTF(char uplo, int n, int kd, double* ab, int ldab)
{
    int info;
    ::F_DPBSTF(&uplo, &n, &kd, ab, &ldab, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBSV computes the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric positive definite band matrix and X
 *  and B are N-by-NRHS matrices.
 *
 *  The Cholesky decomposition is used to factor A as
 *     A = U**T * U,  if UPLO = 'U', or
 *     A = L * L**T,  if UPLO = 'L',
 *  where U is an upper triangular band matrix, and L is a lower
 *  triangular band matrix, with the same number of superdiagonals or
 *  subdiagonals as A.  The factored form of A is then used to solve the
 *  system of equations A * X = B.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(KD+1+i-j,j) = A(i,j) for max(1,j-KD)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(N,j+KD).
 *          See below for further details.
 *
 *          On exit, if INFO = 0, the triangular factor U or L from the
 *          Cholesky factorization A = U**T*U or A = L*L**T of the band
 *          matrix A, in the same storage format as A.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i of A is not
 *                positive definite, so the factorization could not be
 *                completed, and the solution has not been computed.
 *
 *  Further Details
 *  ===============
 *
 *  The band storage scheme is illustrated by the following example, when
 *  N = 6, KD = 2, and UPLO = 'U':
 *
 *  On entry:                       On exit:
 *
 *      *    *   a13  a24  a35  a46      *    *   u13  u24  u35  u46
 *      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
 *     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
 *
 *  Similarly, if UPLO = 'L' the format of A is as follows:
 *
 *  On entry:                       On exit:
 *
 *     a11  a22  a33  a44  a55  a66     l11  l22  l33  l44  l55  l66
 *     a21  a32  a43  a54  a65   *      l21  l32  l43  l54  l65   *
 *     a31  a42  a53  a64   *    *      l31  l42  l53  l64   *    *
 *
 *  Array elements marked * are not used by the routine.
 *
 *  =====================================================================
 *
 *     .. External Functions ..
 **/
int C_DPBSV(char uplo, int n, int kd, int nrhs, double* ab, int ldab, double* b, int ldb)
{
    int info;
    ::F_DPBSV(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 *  compute the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric positive definite band matrix and X
 *  and B are N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'E', real scaling factors are computed to equilibrate
 *     the system:
 *        diag(S) * A * diag(S) * inv(diag(S)) * X = diag(S) * B
 *     Whether or not the system will be equilibrated depends on the
 *     scaling of the matrix A, but if equilibration is used, A is
 *     overwritten by diag(S)*A*diag(S) and B by diag(S)*B.
 *
 *  2. If FACT = 'N' or 'E', the Cholesky decomposition is used to
 *     factor the matrix A (after equilibration if FACT = 'E') as
 *        A = U**T * U,  if UPLO = 'U', or
 *        A = L * L**T,  if UPLO = 'L',
 *     where U is an upper triangular band matrix, and L is a lower
 *     triangular band matrix.
 *
 *  3. If the leading i-by-i principal minor is not positive definite,
 *     then the routine returns with INFO = i. Otherwise, the factored
 *     form of A is used to estimate the condition number of the matrix
 *     A.  If the reciprocal of the condition number is less than machine
 *     precision, INFO = N+1 is returned as a warning, but the routine
 *     still goes on to solve for X and compute error bounds as
 *     described below.
 *
 *  4. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  5. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  6. If equilibration was used, the matrix X is premultiplied by
 *     diag(S) so that it solves the original system before
 *     equilibration.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of the matrix A is
 *          supplied on entry, and if not, whether the matrix A should be
 *          equilibrated before it is factored.
 *          = 'F':  On entry, AFB contains the factored form of A.
 *                  If EQUED = 'Y', the matrix A has been equilibrated
 *                  with scaling factors given by S.  AB and AFB will not
 *                  be modified.
 *          = 'N':  The matrix A will be copied to AFB and factored.
 *          = 'E':  The matrix A will be equilibrated if necessary, then
 *                  copied to AFB and factored.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right-hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array, except
 *          if FACT = 'F' and EQUED = 'Y', then A must contain the
 *          equilibrated matrix diag(S)*A*diag(S).  The j-th column of A
 *          is stored in the j-th column of the array AB as follows:
 *          if UPLO = 'U', AB(KD+1+i-j,j) = A(i,j) for max(1,j-KD)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(N,j+KD).
 *          See below for further details.
 *
 *          On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by
 *          diag(S)*A*diag(S).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array A.  LDAB >= KD+1.
 *
 *  AFB     (input or output) DOUBLE PRECISION array, dimension (LDAFB,N)
 *          If FACT = 'F', then AFB is an input argument and on entry
 *          contains the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T of the band matrix
 *          A, in the same storage format as A (see AB).  If EQUED = 'Y',
 *          then AFB is the factored form of the equilibrated matrix A.
 *
 *          If FACT = 'N', then AFB is an output argument and on exit
 *          returns the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T.
 *
 *          If FACT = 'E', then AFB is an output argument and on exit
 *          returns the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T of the equilibrated
 *          matrix A (see the description of A for the form of the
 *          equilibrated matrix).
 *
 *  LDAFB   (input) INTEGER
 *          The leading dimension of the array AFB.  LDAFB >= KD+1.
 *
 *  EQUED   (input or output) CHARACTER*1
 *          Specifies the form of equilibration that was done.
 *          = 'N':  No equilibration (always true if FACT = 'N').
 *          = 'Y':  Equilibration was done, i.e., A has been replaced by
 *                  diag(S) * A * diag(S).
 *          EQUED is an input argument if FACT = 'F'; otherwise, it is an
 *          output argument.
 *
 *  S       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The scale factors for A; not accessed if EQUED = 'N'.  S is
 *          an input argument if FACT = 'F'; otherwise, S is an output
 *          argument.  If FACT = 'F' and EQUED = 'Y', each element of S
 *          must be positive.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if EQUED = 'N', B is not modified; if EQUED = 'Y',
 *          B is overwritten by diag(S) * B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X to
 *          the original system of equations.  Note that if EQUED = 'Y',
 *          A and B are modified on exit, and the solution to the
 *          equilibrated system is inv(diag(S))*X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A after equilibration (if done).  If RCOND is less than the
 *          machine precision (in particular, if RCOND = 0), the matrix
 *          is singular to working precision.  This condition is
 *          indicated by a return code of INFO > 0.
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  the leading minor of order i of A is
 *                       not positive definite, so the factorization
 *                       could not be completed, and the solution has not
 *                       been computed. RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  Further Details
 *  ===============
 *
 *  The band storage scheme is illustrated by the following example, when
 *  N = 6, KD = 2, and UPLO = 'U':
 *
 *  Two-dimensional storage of the symmetric matrix A:
 *
 *     a11  a12  a13
 *          a22  a23  a24
 *               a33  a34  a35
 *                    a44  a45  a46
 *                         a55  a56
 *     (aij=conjg(aji))         a66
 *
 *  Band storage of the upper triangle of A:
 *
 *      *    *   a13  a24  a35  a46
 *      *   a12  a23  a34  a45  a56
 *     a11  a22  a33  a44  a55  a66
 *
 *  Similarly, if UPLO = 'L' the format of A is as follows:
 *
 *     a11  a22  a33  a44  a55  a66
 *     a21  a32  a43  a54  a65   *
 *     a31  a42  a53  a64   *    *
 *
 *  Array elements marked * are not used by the routine.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPBSVX(char fact, char uplo, int n, int kd, int nrhs, double* ab, int ldab, double* afb, int ldafb, char equed, double* s, double* b, int ldb, double* x, int ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DPBSVX(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, s, b, &ldb, x, &ldx, rcond, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBTRF computes the Cholesky factorization of a real symmetric
 *  positive definite band matrix A.
 *
 *  The factorization has the form
 *     A = U**T * U,  if UPLO = 'U', or
 *     A = L  * L**T,  if UPLO = 'L',
 *  where U is an upper triangular matrix and L is lower triangular.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *          On exit, if INFO = 0, the triangular factor U or L from the
 *          Cholesky factorization A = U**T*U or A = L*L**T of the band
 *          matrix A, in the same storage format as A.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i is not
 *                positive definite, and the factorization could not be
 *                completed.
 *
 *  Further Details
 *  ===============
 *
 *  The band storage scheme is illustrated by the following example, when
 *  N = 6, KD = 2, and UPLO = 'U':
 *
 *  On entry:                       On exit:
 *
 *      *    *   a13  a24  a35  a46      *    *   u13  u24  u35  u46
 *      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
 *     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
 *
 *  Similarly, if UPLO = 'L' the format of A is as follows:
 *
 *  On entry:                       On exit:
 *
 *     a11  a22  a33  a44  a55  a66     l11  l22  l33  l44  l55  l66
 *     a21  a32  a43  a54  a65   *      l21  l32  l43  l54  l65   *
 *     a31  a42  a53  a64   *    *      l31  l42  l53  l64   *    *
 *
 *  Array elements marked * are not used by the routine.
 *
 *  Contributed by
 *  Peter Mayes and Giuseppe Radicati, IBM ECSEC, Rome, March 23, 1989
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPBTRF(char uplo, int n, int kd, double* ab, int ldab)
{
    int info;
    ::F_DPBTRF(&uplo, &n, &kd, ab, &ldab, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPBTRS solves a system of linear equations A*X = B with a symmetric
 *  positive definite band matrix A using the Cholesky factorization
 *  A = U**T*U or A = L*L**T computed by DPBTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangular factor stored in AB;
 *          = 'L':  Lower triangular factor stored in AB.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T of the band matrix A, stored in the
 *          first KD+1 rows of the array.  The j-th column of U or L is
 *          stored in the j-th column of the array AB as follows:
 *          if UPLO ='U', AB(kd+1+i-j,j) = U(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO ='L', AB(1+i-j,j)    = L(i,j) for j<=i<=min(n,j+kd).
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DPBTRS(char uplo, int n, int kd, int nrhs, double* ab, int ldab, double* b, int ldb)
{
    int info;
    ::F_DPBTRS(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOCON estimates the reciprocal of the condition number (in the
 *  1-norm) of a real symmetric positive definite matrix using the
 *  Cholesky factorization A = U**T*U or A = L*L**T computed by DPOTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T, as computed by DPOTRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          The 1-norm (or infinity-norm) of the symmetric matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPOCON(char uplo, int n, double* a, int lda, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DPOCON(&uplo, &n, a, &lda, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOEQU computes row and column scalings intended to equilibrate a
 *  symmetric positive definite matrix A and reduce its condition number
 *  (with respect to the two-norm).  S contains the scale factors,
 *  S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
 *  elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
 *  choice of S puts the condition number of B within a factor N of the
 *  smallest possible condition number over all possible diagonal
 *  scalings.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The N-by-N symmetric positive definite matrix whose scaling
 *          factors are to be computed.  Only the diagonal elements of A
 *          are referenced.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  S       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, S contains the scale factors for A.
 *
 *  SCOND   (output) DOUBLE PRECISION
 *          If INFO = 0, S contains the ratio of the smallest S(i) to
 *          the largest S(i).  If SCOND >= 0.1 and AMAX is neither too
 *          large nor too small, it is not worth scaling by S.
 *
 *  AMAX    (output) DOUBLE PRECISION
 *          Absolute value of largest matrix element.  If AMAX is very
 *          close to overflow or very close to underflow, the matrix
 *          should be scaled.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the i-th diagonal element is nonpositive.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPOEQU(int n, double* a, int lda, double* s, double* scond, double* amax)
{
    int info;
    ::F_DPOEQU(&n, a, &lda, s, scond, amax, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPORFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is symmetric positive definite,
 *  and provides error bounds and backward error estimates for the
 *  solution.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The symmetric matrix A.  If UPLO = 'U', the leading N-by-N
 *          upper triangular part of A contains the upper triangular part
 *          of the matrix A, and the strictly lower triangular part of A
 *          is not referenced.  If UPLO = 'L', the leading N-by-N lower
 *          triangular part of A contains the lower triangular part of
 *          the matrix A, and the strictly upper triangular part of A is
 *          not referenced.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  AF      (input) DOUBLE PRECISION array, dimension (LDAF,N)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T, as computed by DPOTRF.
 *
 *  LDAF    (input) INTEGER
 *          The leading dimension of the array AF.  LDAF >= max(1,N).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DPOTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPORFS(char uplo, int n, int nrhs, double* a, int lda, double* af, int ldaf, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DPORFS(&uplo, &n, &nrhs, a, &lda, af, &ldaf, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOSV computes the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric positive definite matrix and X and B
 *  are N-by-NRHS matrices.
 *
 *  The Cholesky decomposition is used to factor A as
 *     A = U**T* U,  if UPLO = 'U', or
 *     A = L * L**T,  if UPLO = 'L',
 *  where U is an upper triangular matrix and L is a lower triangular
 *  matrix.  The factored form of A is then used to solve the system of
 *  equations A * X = B.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, if INFO = 0, the factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i of A is not
 *                positive definite, so the factorization could not be
 *                completed, and the solution has not been computed.
 *
 *  =====================================================================
 *
 *     .. External Functions ..
 **/
int C_DPOSV(char uplo, int n, int nrhs, double* a, int lda, double* b, int ldb)
{
    int info;
    ::F_DPOSV(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 *  compute the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric positive definite matrix and X and B
 *  are N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'E', real scaling factors are computed to equilibrate
 *     the system:
 *        diag(S) * A * diag(S) * inv(diag(S)) * X = diag(S) * B
 *     Whether or not the system will be equilibrated depends on the
 *     scaling of the matrix A, but if equilibration is used, A is
 *     overwritten by diag(S)*A*diag(S) and B by diag(S)*B.
 *
 *  2. If FACT = 'N' or 'E', the Cholesky decomposition is used to
 *     factor the matrix A (after equilibration if FACT = 'E') as
 *        A = U**T* U,  if UPLO = 'U', or
 *        A = L * L**T,  if UPLO = 'L',
 *     where U is an upper triangular matrix and L is a lower triangular
 *     matrix.
 *
 *  3. If the leading i-by-i principal minor is not positive definite,
 *     then the routine returns with INFO = i. Otherwise, the factored
 *     form of A is used to estimate the condition number of the matrix
 *     A.  If the reciprocal of the condition number is less than machine
 *     precision, INFO = N+1 is returned as a warning, but the routine
 *     still goes on to solve for X and compute error bounds as
 *     described below.
 *
 *  4. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  5. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  6. If equilibration was used, the matrix X is premultiplied by
 *     diag(S) so that it solves the original system before
 *     equilibration.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of the matrix A is
 *          supplied on entry, and if not, whether the matrix A should be
 *          equilibrated before it is factored.
 *          = 'F':  On entry, AF contains the factored form of A.
 *                  If EQUED = 'Y', the matrix A has been equilibrated
 *                  with scaling factors given by S.  A and AF will not
 *                  be modified.
 *          = 'N':  The matrix A will be copied to AF and factored.
 *          = 'E':  The matrix A will be equilibrated if necessary, then
 *                  copied to AF and factored.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A, except if FACT = 'F' and
 *          EQUED = 'Y', then A must contain the equilibrated matrix
 *          diag(S)*A*diag(S).  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.  A is not modified if
 *          FACT = 'F' or 'N', or if FACT = 'E' and EQUED = 'N' on exit.
 *
 *          On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by
 *          diag(S)*A*diag(S).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  AF      (input or output) DOUBLE PRECISION array, dimension (LDAF,N)
 *          If FACT = 'F', then AF is an input argument and on entry
 *          contains the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T, in the same storage
 *          format as A.  If EQUED .ne. 'N', then AF is the factored form
 *          of the equilibrated matrix diag(S)*A*diag(S).
 *
 *          If FACT = 'N', then AF is an output argument and on exit
 *          returns the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T of the original
 *          matrix A.
 *
 *          If FACT = 'E', then AF is an output argument and on exit
 *          returns the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T of the equilibrated
 *          matrix A (see the description of A for the form of the
 *          equilibrated matrix).
 *
 *  LDAF    (input) INTEGER
 *          The leading dimension of the array AF.  LDAF >= max(1,N).
 *
 *  EQUED   (input or output) CHARACTER*1
 *          Specifies the form of equilibration that was done.
 *          = 'N':  No equilibration (always true if FACT = 'N').
 *          = 'Y':  Equilibration was done, i.e., A has been replaced by
 *                  diag(S) * A * diag(S).
 *          EQUED is an input argument if FACT = 'F'; otherwise, it is an
 *          output argument.
 *
 *  S       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The scale factors for A; not accessed if EQUED = 'N'.  S is
 *          an input argument if FACT = 'F'; otherwise, S is an output
 *          argument.  If FACT = 'F' and EQUED = 'Y', each element of S
 *          must be positive.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if EQUED = 'N', B is not modified; if EQUED = 'Y',
 *          B is overwritten by diag(S) * B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X to
 *          the original system of equations.  Note that if EQUED = 'Y',
 *          A and B are modified on exit, and the solution to the
 *          equilibrated system is inv(diag(S))*X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A after equilibration (if done).  If RCOND is less than the
 *          machine precision (in particular, if RCOND = 0), the matrix
 *          is singular to working precision.  This condition is
 *          indicated by a return code of INFO > 0.
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, and i is
 *                <= N:  the leading minor of order i of A is
 *                       not positive definite, so the factorization
 *                       could not be completed, and the solution has not
 *                       been computed. RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPOSVX(char fact, char uplo, int n, int nrhs, double* a, int lda, double* af, int ldaf, char equed, double* s, double* b, int ldb, double* x, int ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DPOSVX(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, &equed, s, b, &ldb, x, &ldx, rcond, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOTRF computes the Cholesky factorization of a real symmetric
 *  positive definite matrix A.
 *
 *  The factorization has the form
 *     A = U**T * U,  if UPLO = 'U', or
 *     A = L  * L**T,  if UPLO = 'L',
 *  where U is an upper triangular matrix and L is lower triangular.
 *
 *  This is the block version of the algorithm, calling Level 3 BLAS.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, if INFO = 0, the factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i is not
 *                positive definite, and the factorization could not be
 *                completed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPOTRF(char uplo, int n, double* a, int lda)
{
    int info;
    ::F_DPOTRF(&uplo, &n, a, &lda, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOTRI computes the inverse of a real symmetric positive definite
 *  matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
 *  computed by DPOTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T, as computed by
 *          DPOTRF.
 *          On exit, the upper or lower triangle of the (symmetric)
 *          inverse of A, overwriting the input factor U or L.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the (i,i) element of the factor U or L is
 *                zero, and the inverse could not be computed.
 *
 *  =====================================================================
 *
 *     .. External Functions ..
 **/
int C_DPOTRI(char uplo, int n, double* a, int lda)
{
    int info;
    ::F_DPOTRI(&uplo, &n, a, &lda, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPOTRS solves a system of linear equations A*X = B with a symmetric
 *  positive definite matrix A using the Cholesky factorization
 *  A = U**T*U or A = L*L**T computed by DPOTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T, as computed by DPOTRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPOTRS(char uplo, int n, int nrhs, double* a, int lda, double* b, int ldb)
{
    int info;
    ::F_DPOTRS(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPCON estimates the reciprocal of the condition number (in the
 *  1-norm) of a real symmetric positive definite packed matrix using
 *  the Cholesky factorization A = U**T*U or A = L*L**T computed by
 *  DPPTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T, packed columnwise in a linear
 *          array.  The j-th column of U or L is stored in the array AP
 *          as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = U(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = L(i,j) for j<=i<=n.
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          The 1-norm (or infinity-norm) of the symmetric matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPPCON(char uplo, int n, double* ap, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DPPCON(&uplo, &n, ap, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPEQU computes row and column scalings intended to equilibrate a
 *  symmetric positive definite matrix A in packed storage and reduce
 *  its condition number (with respect to the two-norm).  S contains the
 *  scale factors, S(i)=1/sqrt(A(i,i)), chosen so that the scaled matrix
 *  B with elements B(i,j)=S(i)*A(i,j)*S(j) has ones on the diagonal.
 *  This choice of S puts the condition number of B within a factor N of
 *  the smallest possible condition number over all possible diagonal
 *  scalings.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangle of the symmetric matrix A, packed
 *          columnwise in a linear array.  The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *
 *  S       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, S contains the scale factors for A.
 *
 *  SCOND   (output) DOUBLE PRECISION
 *          If INFO = 0, S contains the ratio of the smallest S(i) to
 *          the largest S(i).  If SCOND >= 0.1 and AMAX is neither too
 *          large nor too small, it is not worth scaling by S.
 *
 *  AMAX    (output) DOUBLE PRECISION
 *          Absolute value of largest matrix element.  If AMAX is very
 *          close to overflow or very close to underflow, the matrix
 *          should be scaled.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the i-th diagonal element is nonpositive.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPPEQU(char uplo, int n, double* ap, double* s, double* scond, double* amax)
{
    int info;
    ::F_DPPEQU(&uplo, &n, ap, s, scond, amax, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is symmetric positive definite
 *  and packed, and provides error bounds and backward error estimates
 *  for the solution.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangle of the symmetric matrix A, packed
 *          columnwise in a linear array.  The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *
 *  AFP     (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T, as computed by DPPTRF/ZPPTRF,
 *          packed columnwise in a linear array in the same format as A
 *          (see AP).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DPPTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPPRFS(char uplo, int n, int nrhs, double* ap, double* afp, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DPPRFS(&uplo, &n, &nrhs, ap, afp, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPSV computes the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric positive definite matrix stored in
 *  packed format and X and B are N-by-NRHS matrices.
 *
 *  The Cholesky decomposition is used to factor A as
 *     A = U**T* U,  if UPLO = 'U', or
 *     A = L * L**T,  if UPLO = 'L',
 *  where U is an upper triangular matrix and L is a lower triangular
 *  matrix.  The factored form of A is then used to solve the system of
 *  equations A * X = B.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *          See below for further details.
 *
 *          On exit, if INFO = 0, the factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T, in the same storage
 *          format as A.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i of A is not
 *                positive definite, so the factorization could not be
 *                completed, and the solution has not been computed.
 *
 *  Further Details
 *  ===============
 *
 *  The packed storage scheme is illustrated by the following example
 *  when N = 4, UPLO = 'U':
 *
 *  Two-dimensional storage of the symmetric matrix A:
 *
 *     a11 a12 a13 a14
 *         a22 a23 a24
 *             a33 a34     (aij = conjg(aji))
 *                 a44
 *
 *  Packed storage of the upper triangle of A:
 *
 *  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
 *
 *  =====================================================================
 *
 *     .. External Functions ..
 **/
int C_DPPSV(char uplo, int n, int nrhs, double* ap, double* b, int ldb)
{
    int info;
    ::F_DPPSV(&uplo, &n, &nrhs, ap, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 *  compute the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric positive definite matrix stored in
 *  packed format and X and B are N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'E', real scaling factors are computed to equilibrate
 *     the system:
 *        diag(S) * A * diag(S) * inv(diag(S)) * X = diag(S) * B
 *     Whether or not the system will be equilibrated depends on the
 *     scaling of the matrix A, but if equilibration is used, A is
 *     overwritten by diag(S)*A*diag(S) and B by diag(S)*B.
 *
 *  2. If FACT = 'N' or 'E', the Cholesky decomposition is used to
 *     factor the matrix A (after equilibration if FACT = 'E') as
 *        A = U**T* U,  if UPLO = 'U', or
 *        A = L * L**T,  if UPLO = 'L',
 *     where U is an upper triangular matrix and L is a lower triangular
 *     matrix.
 *
 *  3. If the leading i-by-i principal minor is not positive definite,
 *     then the routine returns with INFO = i. Otherwise, the factored
 *     form of A is used to estimate the condition number of the matrix
 *     A.  If the reciprocal of the condition number is less than machine
 *     precision, INFO = N+1 is returned as a warning, but the routine
 *     still goes on to solve for X and compute error bounds as
 *     described below.
 *
 *  4. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  5. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  6. If equilibration was used, the matrix X is premultiplied by
 *     diag(S) so that it solves the original system before
 *     equilibration.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of the matrix A is
 *          supplied on entry, and if not, whether the matrix A should be
 *          equilibrated before it is factored.
 *          = 'F':  On entry, AFP contains the factored form of A.
 *                  If EQUED = 'Y', the matrix A has been equilibrated
 *                  with scaling factors given by S.  AP and AFP will not
 *                  be modified.
 *          = 'N':  The matrix A will be copied to AFP and factored.
 *          = 'E':  The matrix A will be equilibrated if necessary, then
 *                  copied to AFP and factored.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array, except if FACT = 'F'
 *          and EQUED = 'Y', then A must contain the equilibrated matrix
 *          diag(S)*A*diag(S).  The j-th column of A is stored in the
 *          array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *          See below for further details.  A is not modified if
 *          FACT = 'F' or 'N', or if FACT = 'E' and EQUED = 'N' on exit.
 *
 *          On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by
 *          diag(S)*A*diag(S).
 *
 *  AFP     (input or output) DOUBLE PRECISION array, dimension
 *                            (N*(N+1)/2)
 *          If FACT = 'F', then AFP is an input argument and on entry
 *          contains the triangular factor U or L from the Cholesky
 *          factorization A = U'*U or A = L*L', in the same storage
 *          format as A.  If EQUED .ne. 'N', then AFP is the factored
 *          form of the equilibrated matrix A.
 *
 *          If FACT = 'N', then AFP is an output argument and on exit
 *          returns the triangular factor U or L from the Cholesky
 *          factorization A = U'*U or A = L*L' of the original matrix A.
 *
 *          If FACT = 'E', then AFP is an output argument and on exit
 *          returns the triangular factor U or L from the Cholesky
 *          factorization A = U'*U or A = L*L' of the equilibrated
 *          matrix A (see the description of AP for the form of the
 *          equilibrated matrix).
 *
 *  EQUED   (input or output) CHARACTER*1
 *          Specifies the form of equilibration that was done.
 *          = 'N':  No equilibration (always true if FACT = 'N').
 *          = 'Y':  Equilibration was done, i.e., A has been replaced by
 *                  diag(S) * A * diag(S).
 *          EQUED is an input argument if FACT = 'F'; otherwise, it is an
 *          output argument.
 *
 *  S       (input or output) DOUBLE PRECISION array, dimension (N)
 *          The scale factors for A; not accessed if EQUED = 'N'.  S is
 *          an input argument if FACT = 'F'; otherwise, S is an output
 *          argument.  If FACT = 'F' and EQUED = 'Y', each element of S
 *          must be positive.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if EQUED = 'N', B is not modified; if EQUED = 'Y',
 *          B is overwritten by diag(S) * B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X to
 *          the original system of equations.  Note that if EQUED = 'Y',
 *          A and B are modified on exit, and the solution to the
 *          equilibrated system is inv(diag(S))*X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A after equilibration (if done).  If RCOND is less than the
 *          machine precision (in particular, if RCOND = 0), the matrix
 *          is singular to working precision.  This condition is
 *          indicated by a return code of INFO > 0.
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  the leading minor of order i of A is
 *                       not positive definite, so the factorization
 *                       could not be completed, and the solution has not
 *                       been computed. RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  Further Details
 *  ===============
 *
 *  The packed storage scheme is illustrated by the following example
 *  when N = 4, UPLO = 'U':
 *
 *  Two-dimensional storage of the symmetric matrix A:
 *
 *     a11 a12 a13 a14
 *         a22 a23 a24
 *             a33 a34     (aij = conjg(aji))
 *                 a44
 *
 *  Packed storage of the upper triangle of A:
 *
 *  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPPSVX(char fact, char uplo, int n, int nrhs, double* ap, double* afp, char equed, double* s, double* b, int ldb, double* x, int ldx, double* rcond, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DPPSVX(&fact, &uplo, &n, &nrhs, ap, afp, &equed, s, b, &ldb, x, &ldx, rcond, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPTRF computes the Cholesky factorization of a real symmetric
 *  positive definite matrix A stored in packed format.
 *
 *  The factorization has the form
 *     A = U**T * U,  if UPLO = 'U', or
 *     A = L  * L**T,  if UPLO = 'L',
 *  where U is an upper triangular matrix and L is lower triangular.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *          See below for further details.
 *
 *          On exit, if INFO = 0, the triangular factor U or L from the
 *          Cholesky factorization A = U**T*U or A = L*L**T, in the same
 *          storage format as A.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i is not
 *                positive definite, and the factorization could not be
 *                completed.
 *
 *  Further Details
 *  ======= =======
 *
 *  The packed storage scheme is illustrated by the following example
 *  when N = 4, UPLO = 'U':
 *
 *  Two-dimensional storage of the symmetric matrix A:
 *
 *     a11 a12 a13 a14
 *         a22 a23 a24
 *             a33 a34     (aij = aji)
 *                 a44
 *
 *  Packed storage of the upper triangle of A:
 *
 *  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPPTRF(char uplo, int n, double* ap)
{
    int info;
    ::F_DPPTRF(&uplo, &n, ap, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPTRI computes the inverse of a real symmetric positive definite
 *  matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
 *  computed by DPPTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangular factor is stored in AP;
 *          = 'L':  Lower triangular factor is stored in AP.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the triangular factor U or L from the Cholesky
 *          factorization A = U**T*U or A = L*L**T, packed columnwise as
 *          a linear array.  The j-th column of U or L is stored in the
 *          array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = U(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = L(i,j) for j<=i<=n.
 *
 *          On exit, the upper or lower triangle of the (symmetric)
 *          inverse of A, overwriting the input factor U or L.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the (i,i) element of the factor U or L is
 *                zero, and the inverse could not be computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPPTRI(char uplo, int n, double* ap)
{
    int info;
    ::F_DPPTRI(&uplo, &n, ap, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPPTRS solves a system of linear equations A*X = B with a symmetric
 *  positive definite matrix A in packed storage using the Cholesky
 *  factorization A = U**T*U or A = L*L**T computed by DPPTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U**T*U or A = L*L**T, packed columnwise in a linear
 *          array.  The j-th column of U or L is stored in the array AP
 *          as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = U(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = L(i,j) for j<=i<=n.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DPPTRS(char uplo, int n, int nrhs, double* ap, double* b, int ldb)
{
    int info;
    ::F_DPPTRS(&uplo, &n, &nrhs, ap, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTCON computes the reciprocal of the condition number (in the
 *  1-norm) of a real symmetric positive definite tridiagonal matrix
 *  using the factorization A = L*D*L**T or A = U**T*D*U computed by
 *  DPTTRF.
 *
 *  Norm(inv(A)) is computed by a direct method, and the reciprocal of
 *  the condition number is computed as
 *               RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the diagonal matrix D from the
 *          factorization of A, as computed by DPTTRF.
 *
 *  E       (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) off-diagonal elements of the unit bidiagonal factor
 *          U or L from the factorization of A,  as computed by DPTTRF.
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          The 1-norm of the original matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is the
 *          1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The method used is described in Nicholas J. Higham, "Efficient
 *  Algorithms for Computing the Condition Number of a Tridiagonal
 *  Matrix", SIAM J. Sci. Stat. Comput., Vol. 7, No. 1, January 1986.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPTCON(int n, double* d, double* e, double anorm, double* rcond, double* work)
{
    int info;
    ::F_DPTCON(&n, d, e, &anorm, rcond, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTEQR computes all eigenvalues and, optionally, eigenvectors of a
 *  symmetric positive definite tridiagonal matrix by first factoring the
 *  matrix using DPTTRF, and then calling DBDSQR to compute the singular
 *  values of the bidiagonal factor.
 *
 *  This routine computes the eigenvalues of the positive definite
 *  tridiagonal matrix to high relative accuracy.  This means that if the
 *  eigenvalues range over many orders of magnitude in size, then the
 *  small eigenvalues and corresponding eigenvectors will be computed
 *  more accurately than, for example, with the standard QR method.
 *
 *  The eigenvectors of a full or band symmetric positive definite matrix
 *  can also be found if DSYTRD, DSPTRD, or DSBTRD has been used to
 *  reduce this matrix to tridiagonal form. (The reduction to tridiagonal
 *  form, however, may preclude the possibility of obtaining high
 *  relative accuracy in the small eigenvalues of the original matrix, if
 *  these eigenvalues range over many orders of magnitude.)
 *
 *  Arguments
 *  =========
 *
 *  COMPZ   (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only.
 *          = 'V':  Compute eigenvectors of original symmetric
 *                  matrix also.  Array Z contains the orthogonal
 *                  matrix used to reduce the original matrix to
 *                  tridiagonal form.
 *          = 'I':  Compute eigenvectors of tridiagonal matrix also.
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal
 *          matrix.
 *          On normal exit, D contains the eigenvalues, in descending
 *          order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix.
 *          On exit, E has been destroyed.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          On entry, if COMPZ = 'V', the orthogonal matrix used in the
 *          reduction to tridiagonal form.
 *          On exit, if COMPZ = 'V', the orthonormal eigenvectors of the
 *          original symmetric matrix;
 *          if COMPZ = 'I', the orthonormal eigenvectors of the
 *          tridiagonal matrix.
 *          If INFO > 0 on exit, Z contains the eigenvectors associated
 *          with only the stored eigenvalues.
 *          If  COMPZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          COMPZ = 'V' or 'I', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (4*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, and i is:
 *                <= N  the Cholesky factorization of the matrix could
 *                      not be performed because the i-th principal minor
 *                      was not positive definite.
 *                > N   the SVD algorithm failed to converge;
 *                      if INFO = N+i, i off-diagonal elements of the
 *                      bidiagonal factor did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPTEQR(char compz, int n, double* d, double* e, double* z, int ldz, double* work)
{
    int info;
    ::F_DPTEQR(&compz, &n, d, e, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is symmetric positive definite
 *  and tridiagonal, and provides error bounds and backward error
 *  estimates for the solution.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the tridiagonal matrix A.
 *
 *  E       (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of the tridiagonal matrix A.
 *
 *  DF      (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the diagonal matrix D from the
 *          factorization computed by DPTTRF.
 *
 *  EF      (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of the unit bidiagonal factor
 *          L from the factorization computed by DPTTRF.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DPTTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPTRFS(int n, int nrhs, double* d, double* e, double* df, double* ef, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work)
{
    int info;
    ::F_DPTRFS(&n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, ferr, berr, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTSV computes the solution to a real system of linear equations
 *  A*X = B, where A is an N-by-N symmetric positive definite tridiagonal
 *  matrix, and X and B are N-by-NRHS matrices.
 *
 *  A is factored as A = L*D*L**T, and the factored form of A is then
 *  used to solve the system of equations.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix
 *          A.  On exit, the n diagonal elements of the diagonal matrix
 *          D from the factorization A = L*D*L**T.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix A.  On exit, the (n-1) subdiagonal elements of the
 *          unit bidiagonal factor L from the L*D*L**T factorization of
 *          A.  (E can also be regarded as the superdiagonal of the unit
 *          bidiagonal factor U from the U**T*D*U factorization of A.)
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the leading minor of order i is not
 *                positive definite, and the solution has not been
 *                computed.  The factorization has not been completed
 *                unless i = N.
 *
 *  =====================================================================
 *
 *     .. External Subroutines ..
 **/
int C_DPTSV(int n, int nrhs, double* d, double* e, double* b, int ldb)
{
    int info;
    ::F_DPTSV(&n, &nrhs, d, e, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTSVX uses the factorization A = L*D*L**T to compute the solution
 *  to a real system of linear equations A*X = B, where A is an N-by-N
 *  symmetric positive definite tridiagonal matrix and X and B are
 *  N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'N', the matrix A is factored as A = L*D*L**T, where L
 *     is a unit lower bidiagonal matrix and D is diagonal.  The
 *     factorization can also be regarded as having the form
 *     A = U**T*D*U.
 *
 *  2. If the leading i-by-i principal minor is not positive definite,
 *     then the routine returns with INFO = i. Otherwise, the factored
 *     form of A is used to estimate the condition number of the matrix
 *     A.  If the reciprocal of the condition number is less than machine
 *     precision, INFO = N+1 is returned as a warning, but the routine
 *     still goes on to solve for X and compute error bounds as
 *     described below.
 *
 *  3. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  4. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of A has been
 *          supplied on entry.
 *          = 'F':  On entry, DF and EF contain the factored form of A.
 *                  D, E, DF, and EF will not be modified.
 *          = 'N':  The matrix A will be copied to DF and EF and
 *                  factored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the tridiagonal matrix A.
 *
 *  E       (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of the tridiagonal matrix A.
 *
 *  DF      (input or output) DOUBLE PRECISION array, dimension (N)
 *          If FACT = 'F', then DF is an input argument and on entry
 *          contains the n diagonal elements of the diagonal matrix D
 *          from the L*D*L**T factorization of A.
 *          If FACT = 'N', then DF is an output argument and on exit
 *          contains the n diagonal elements of the diagonal matrix D
 *          from the L*D*L**T factorization of A.
 *
 *  EF      (input or output) DOUBLE PRECISION array, dimension (N-1)
 *          If FACT = 'F', then EF is an input argument and on entry
 *          contains the (n-1) subdiagonal elements of the unit
 *          bidiagonal factor L from the L*D*L**T factorization of A.
 *          If FACT = 'N', then EF is an output argument and on exit
 *          contains the (n-1) subdiagonal elements of the unit
 *          bidiagonal factor L from the L*D*L**T factorization of A.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The N-by-NRHS right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 of INFO = N+1, the N-by-NRHS solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal condition number of the matrix A.  If RCOND
 *          is less than the machine precision (in particular, if
 *          RCOND = 0), the matrix is singular to working precision.
 *          This condition is indicated by a return code of INFO > 0.
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in any
 *          element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  the leading minor of order i of A is
 *                       not positive definite, so the factorization
 *                       could not be completed, and the solution has not
 *                       been computed. RCOND = 0 is returned.
 *                = N+1: U is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPTSVX(char fact, int n, int nrhs, double* d, double* e, double* df, double* ef, double* b, int ldb, double* x, int ldx, double* rcond, double* ferr, double* berr, double* work)
{
    int info;
    ::F_DPTSVX(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, rcond, ferr, berr, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTTRF computes the L*D*L' factorization of a real symmetric
 *  positive definite tridiagonal matrix A.  The factorization may also
 *  be regarded as having the form A = U'*D*U.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix
 *          A.  On exit, the n diagonal elements of the diagonal matrix
 *          D from the L*D*L' factorization of A.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix A.  On exit, the (n-1) subdiagonal elements of the
 *          unit bidiagonal factor L from the L*D*L' factorization of A.
 *          E can also be regarded as the superdiagonal of the unit
 *          bidiagonal factor U from the U'*D*U factorization of A.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -k, the k-th argument had an illegal value
 *          > 0: if INFO = k, the leading minor of order k is not
 *               positive definite; if k < N, the factorization could not
 *               be completed, while if k = N, the factorization was
 *               completed, but D(N) <= 0.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DPTTRF(int n, double* d, double* e)
{
    int info;
    ::F_DPTTRF(&n, d, e, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DPTTRS solves a tridiagonal system of the form
 *     A * X = B
 *  using the L*D*L' factorization of A computed by DPTTRF.  D is a
 *  diagonal matrix specified in the vector D, L is a unit bidiagonal
 *  matrix whose subdiagonal is specified in the vector E, and X and B
 *  are N by NRHS matrices.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the tridiagonal matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the diagonal matrix D from the
 *          L*D*L' factorization of A.
 *
 *  E       (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of the unit bidiagonal factor
 *          L from the L*D*L' factorization of A.  E can also be regarded
 *          as the superdiagonal of the unit bidiagonal factor U from the
 *          factorization A = U'*D*U.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side vectors B for the system of
 *          linear equations.
 *          On exit, the solution vectors, X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -k, the k-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DPTTRS(int n, int nrhs, double* d, double* e, double* b, int ldb)
{
    int info;
    ::F_DPTTRS(&n, &nrhs, d, e, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBEV computes all the eigenvalues and, optionally, eigenvectors of
 *  a real symmetric band matrix A.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *          On exit, AB is overwritten by values generated during the
 *          reduction to tridiagonal form.  If UPLO = 'U', the first
 *          superdiagonal and the diagonal of the tridiagonal matrix T
 *          are returned in rows KD and KD+1 of AB, and if UPLO = 'L',
 *          the diagonal and first subdiagonal of T are returned in the
 *          first two rows of AB.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD + 1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
 *          eigenvectors of the matrix A, with the i-th column of Z
 *          holding the eigenvector associated with W(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (max(1,3*N-2))
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the algorithm failed to converge; i
 *                off-diagonal elements of an intermediate tridiagonal
 *                form did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBEV(char jobz, char uplo, int n, int kd, double* ab, int ldab, double* w, double* z, int ldz, double* work)
{
    int info;
    ::F_DSBEV(&jobz, &uplo, &n, &kd, ab, &ldab, w, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBEVD computes all the eigenvalues and, optionally, eigenvectors of
 *  a real symmetric band matrix A. If eigenvectors are desired, it uses
 *  a divide and conquer algorithm.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *          On exit, AB is overwritten by values generated during the
 *          reduction to tridiagonal form.  If UPLO = 'U', the first
 *          superdiagonal and the diagonal of the tridiagonal matrix T
 *          are returned in rows KD and KD+1 of AB, and if UPLO = 'L',
 *          the diagonal and first subdiagonal of T are returned in the
 *          first two rows of AB.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD + 1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
 *          eigenvectors of the matrix A, with the i-th column of Z
 *          holding the eigenvector associated with W(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array,
 *                                         dimension (LWORK)
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          IF N <= 1,                LWORK must be at least 1.
 *          If JOBZ  = 'N' and N > 2, LWORK must be at least 2*N.
 *          If JOBZ  = 'V' and N > 2, LWORK must be at least
 *                         ( 1 + 5*N + 2*N**2 ).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array LIWORK.
 *          If JOBZ  = 'N' or N <= 1, LIWORK must be at least 1.
 *          If JOBZ  = 'V' and N > 2, LIWORK must be at least 3 + 5*N.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the algorithm failed to converge; i
 *                off-diagonal elements of an intermediate tridiagonal
 *                form did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBEVD(char jobz, char uplo, int n, int kd, double* ab, int ldab, double* w, double* z, int ldz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSBEVD(&jobz, &uplo, &n, &kd, ab, &ldab, w, z, &ldz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBEVX computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric band matrix A.  Eigenvalues and eigenvectors can
 *  be selected by specifying either a range of values or a range of
 *  indices for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found;
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found;
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *
 *          On exit, AB is overwritten by values generated during the
 *          reduction to tridiagonal form.  If UPLO = 'U', the first
 *          superdiagonal and the diagonal of the tridiagonal matrix T
 *          are returned in rows KD and KD+1 of AB, and if UPLO = 'L',
 *          the diagonal and first subdiagonal of T are returned in the
 *          first two rows of AB.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD + 1.
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ, N)
 *          If JOBZ = 'V', the N-by-N orthogonal matrix used in the
 *                         reduction to tridiagonal form.
 *          If JOBZ = 'N', the array Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.  If JOBZ = 'V', then
 *          LDQ >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing AB to tridiagonal form.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *          See "Computing Small Singular Values of Bidiagonal Matrices
 *          with Guaranteed High Relative Accuracy," by Demmel and
 *          Kahan, LAPACK Working Note #3.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          The first M elements contain the selected eigenvalues in
 *          ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          If an eigenvector fails to converge, then that column of Z
 *          contains the latest approximation to the eigenvector, and the
 *          index of the eigenvector is returned in IFAIL.
 *          If JOBZ = 'N', then Z is not referenced.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (7*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (N)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvectors that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, then i eigenvectors failed to converge.
 *                Their indices are stored in array IFAIL.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBEVX(char jobz, char range, char uplo, int n, int kd, double* ab, int ldab, double* q, int ldq, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int* iwork, int* ifail)
{
    int info;
    ::F_DSBEVX(&jobz, &range, &uplo, &n, &kd, ab, &ldab, q, &ldq, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBGST reduces a real symmetric-definite banded generalized
 *  eigenproblem  A*x = lambda*B*x  to standard form  C*y = lambda*y,
 *  such that C has the same bandwidth as A.
 *
 *  B must have been previously factorized as S**T*S by DPBSTF, using a
 *  split Cholesky factorization. A is overwritten by C = X**T*A*X, where
 *  X = S**(-1)*Q and Q is an orthogonal matrix chosen to preserve the
 *  bandwidth of A.
 *
 *  Arguments
 *  =========
 *
 *  VECT    (input) CHARACTER*1
 *          = 'N':  do not form the transformation matrix X;
 *          = 'V':  form X.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  KA      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KA >= 0.
 *
 *  KB      (input) INTEGER
 *          The number of superdiagonals of the matrix B if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KA >= KB >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first ka+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka).
 *
 *          On exit, the transformed matrix X**T*A*X, stored in the same
 *          format as A.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KA+1.
 *
 *  BB      (input) DOUBLE PRECISION array, dimension (LDBB,N)
 *          The banded factor S from the split Cholesky factorization of
 *          B, as returned by DPBSTF, stored in the first KB+1 rows of
 *          the array.
 *
 *  LDBB    (input) INTEGER
 *          The leading dimension of the array BB.  LDBB >= KB+1.
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,N)
 *          If VECT = 'V', the n-by-n matrix X.
 *          If VECT = 'N', the array X is not referenced.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.
 *          LDX >= max(1,N) if VECT = 'V'; LDX >= 1 otherwise.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBGST(char vect, char uplo, int n, int ka, int kb, double* ab, int ldab, double* bb, int ldbb, double* x, int ldx, double* work)
{
    int info;
    ::F_DSBGST(&vect, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, x, &ldx, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBGV computes all the eigenvalues, and optionally, the eigenvectors
 *  of a real generalized symmetric-definite banded eigenproblem, of
 *  the form A*x=(lambda)*B*x. Here A and B are assumed to be symmetric
 *  and banded, and B is also positive definite.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  KA      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'. KA >= 0.
 *
 *  KB      (input) INTEGER
 *          The number of superdiagonals of the matrix B if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'. KB >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first ka+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka).
 *
 *          On exit, the contents of AB are destroyed.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KA+1.
 *
 *  BB      (input/output) DOUBLE PRECISION array, dimension (LDBB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix B, stored in the first kb+1 rows of the array.  The
 *          j-th column of B is stored in the j-th column of the array BB
 *          as follows:
 *          if UPLO = 'U', BB(kb+1+i-j,j) = B(i,j) for max(1,j-kb)<=i<=j;
 *          if UPLO = 'L', BB(1+i-j,j)    = B(i,j) for j<=i<=min(n,j+kb).
 *
 *          On exit, the factor S from the split Cholesky factorization
 *          B = S**T*S, as returned by DPBSTF.
 *
 *  LDBB    (input) INTEGER
 *          The leading dimension of the array BB.  LDBB >= KB+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of
 *          eigenvectors, with the i-th column of Z holding the
 *          eigenvector associated with W(i). The eigenvectors are
 *          normalized so that Z**T*B*Z = I.
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= N.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is:
 *             <= N:  the algorithm failed to converge:
 *                    i off-diagonal elements of an intermediate
 *                    tridiagonal form did not converge to zero;
 *             > N:   if INFO = N + i, for 1 <= i <= N, then DPBSTF
 *                    returned INFO = i: B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DSBGV(char jobz, char uplo, int n, int ka, int kb, double* ab, int ldab, double* bb, int ldbb, double* w, double* z, int ldz, double* work)
{
    int info;
    ::F_DSBGV(&jobz, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, w, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBGVD computes all the eigenvalues, and optionally, the eigenvectors
 *  of a real generalized symmetric-definite banded eigenproblem, of the
 *  form A*x=(lambda)*B*x.  Here A and B are assumed to be symmetric and
 *  banded, and B is also positive definite.  If eigenvectors are
 *  desired, it uses a divide and conquer algorithm.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  KA      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KA >= 0.
 *
 *  KB      (input) INTEGER
 *          The number of superdiagonals of the matrix B if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KB >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first ka+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka).
 *
 *          On exit, the contents of AB are destroyed.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KA+1.
 *
 *  BB      (input/output) DOUBLE PRECISION array, dimension (LDBB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix B, stored in the first kb+1 rows of the array.  The
 *          j-th column of B is stored in the j-th column of the array BB
 *          as follows:
 *          if UPLO = 'U', BB(ka+1+i-j,j) = B(i,j) for max(1,j-kb)<=i<=j;
 *          if UPLO = 'L', BB(1+i-j,j)    = B(i,j) for j<=i<=min(n,j+kb).
 *
 *          On exit, the factor S from the split Cholesky factorization
 *          B = S**T*S, as returned by DPBSTF.
 *
 *  LDBB    (input) INTEGER
 *          The leading dimension of the array BB.  LDBB >= KB+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of
 *          eigenvectors, with the i-th column of Z holding the
 *          eigenvector associated with W(i).  The eigenvectors are
 *          normalized so Z**T*B*Z = I.
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If N <= 1,               LWORK >= 1.
 *          If JOBZ = 'N' and N > 1, LWORK >= 3*N.
 *          If JOBZ = 'V' and N > 1, LWORK >= 1 + 5*N + 2*N**2.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if LIWORK > 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If JOBZ  = 'N' or N <= 1, LIWORK >= 1.
 *          If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is:
 *             <= N:  the algorithm failed to converge:
 *                    i off-diagonal elements of an intermediate
 *                    tridiagonal form did not converge to zero;
 *             > N:   if INFO = N + i, for 1 <= i <= N, then DPBSTF
 *                    returned INFO = i: B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBGVD(char jobz, char uplo, int n, int ka, int kb, double* ab, int ldab, double* bb, int ldbb, double* w, double* z, int ldz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSBGVD(&jobz, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, w, z, &ldz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBGVX computes selected eigenvalues, and optionally, eigenvectors
 *  of a real generalized symmetric-definite banded eigenproblem, of
 *  the form A*x=(lambda)*B*x.  Here A and B are assumed to be symmetric
 *  and banded, and B is also positive definite.  Eigenvalues and
 *  eigenvectors can be selected by specifying either all eigenvalues,
 *  a range of values or a range of indices for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  KA      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KA >= 0.
 *
 *  KB      (input) INTEGER
 *          The number of superdiagonals of the matrix B if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KB >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first ka+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka).
 *
 *          On exit, the contents of AB are destroyed.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KA+1.
 *
 *  BB      (input/output) DOUBLE PRECISION array, dimension (LDBB, N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix B, stored in the first kb+1 rows of the array.  The
 *          j-th column of B is stored in the j-th column of the array BB
 *          as follows:
 *          if UPLO = 'U', BB(ka+1+i-j,j) = B(i,j) for max(1,j-kb)<=i<=j;
 *          if UPLO = 'L', BB(1+i-j,j)    = B(i,j) for j<=i<=min(n,j+kb).
 *
 *          On exit, the factor S from the split Cholesky factorization
 *          B = S**T*S, as returned by DPBSTF.
 *
 *  LDBB    (input) INTEGER
 *          The leading dimension of the array BB.  LDBB >= KB+1.
 *
 *  Q       (output) DOUBLE PRECISION array, dimension (LDQ, N)
 *          If JOBZ = 'V', the n-by-n matrix used in the reduction of
 *          A*x = (lambda)*B*x to standard form, i.e. C*x = (lambda)*x,
 *          and consequently C to tridiagonal form.
 *          If JOBZ = 'N', the array Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.  If JOBZ = 'N',
 *          LDQ >= 1. If JOBZ = 'V', LDQ >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing A to tridiagonal form.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of
 *          eigenvectors, with the i-th column of Z holding the
 *          eigenvector associated with W(i).  The eigenvectors are
 *          normalized so Z**T*B*Z = I.
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (7*N)
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (M)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvalues that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0 : successful exit
 *          < 0 : if INFO = -i, the i-th argument had an illegal value
 *          <= N: if INFO = i, then i eigenvectors failed to converge.
 *                  Their indices are stored in IFAIL.
 *          > N : DPBSTF returned an error code; i.e.,
 *                if INFO = N + i, for 1 <= i <= N, then the leading
 *                minor of order i of B is not positive definite.
 *                The factorization of B could not be completed and
 *                no eigenvalues or eigenvectors were computed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBGVX(char jobz, char range, char uplo, int n, int ka, int kb, double* ab, int ldab, double* bb, int ldbb, double* q, int ldq, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int* iwork, int* ifail)
{
    int info;
    ::F_DSBGVX(&jobz, &range, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, q, &ldq, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSBTRD reduces a real symmetric band matrix A to symmetric
 *  tridiagonal form T by an orthogonal similarity transformation:
 *  Q**T * A * Q = T.
 *
 *  Arguments
 *  =========
 *
 *  VECT    (input) CHARACTER*1
 *          = 'N':  do not form Q;
 *          = 'V':  form Q;
 *          = 'U':  update a matrix X, by forming X*Q.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals of the matrix A if UPLO = 'U',
 *          or the number of subdiagonals if UPLO = 'L'.  KD >= 0.
 *
 *  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N)
 *          On entry, the upper or lower triangle of the symmetric band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *          On exit, the diagonal elements of AB are overwritten by the
 *          diagonal elements of the tridiagonal matrix T; if KD > 0, the
 *          elements on the first superdiagonal (if UPLO = 'U') or the
 *          first subdiagonal (if UPLO = 'L') are overwritten by the
 *          off-diagonal elements of T; the rest of AB is overwritten by
 *          values generated during the reduction.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  D       (output) DOUBLE PRECISION array, dimension (N)
 *          The diagonal elements of the tridiagonal matrix T.
 *
 *  E       (output) DOUBLE PRECISION array, dimension (N-1)
 *          The off-diagonal elements of the tridiagonal matrix T:
 *          E(i) = T(i,i+1) if UPLO = 'U'; E(i) = T(i+1,i) if UPLO = 'L'.
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          On entry, if VECT = 'U', then Q must contain an N-by-N
 *          matrix X; if VECT = 'N' or 'V', then Q need not be set.
 *
 *          On exit:
 *          if VECT = 'V', Q contains the N-by-N orthogonal matrix Q;
 *          if VECT = 'U', Q contains the product X*Q;
 *          if VECT = 'N', the array Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.
 *          LDQ >= 1, and LDQ >= N if VECT = 'V' or 'U'.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  Modified by Linda Kaufman, Bell Labs.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSBTRD(char vect, char uplo, int n, int kd, double* ab, int ldab, double* d, double* e, double* q, int ldq, double* work)
{
    int info;
    ::F_DSBTRD(&vect, &uplo, &n, &kd, ab, &ldab, d, e, q, &ldq, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSGESV computes the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
 *
 *  DSGESV first attempts to factorize the matrix in SINGLE PRECISION
 *  and use this factorization within an iterative refinement procedure
 *  to produce a solution with DOUBLE PRECISION normwise backward error
 *  quality (see below). If the approach fails the method switches to a
 *  DOUBLE PRECISION factorization and solve.
 *
 *  The iterative refinement is not going to be a winning strategy if
 *  the ratio SINGLE PRECISION performance over DOUBLE PRECISION
 *  performance is too small. A reasonable strategy should take the
 *  number of right-hand sides and the size of the matrix into account.
 *  This might be done with a call to ILAENV in the future. Up to now, we
 *  always try iterative refinement.
 *
 *  The iterative refinement process is stopped if
 *      ITER > ITERMAX
 *  or for all the RHS we have:
 *      RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
 *  where
 *      o ITER is the number of the current iteration in the iterative
 *        refinement process
 *      o RNRM is the infinity-norm of the residual
 *      o XNRM is the infinity-norm of the solution
 *      o ANRM is the infinity-operator-norm of the matrix A
 *      o EPS is the machine epsilon returned by DLAMCH('Epsilon')
 *  The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00
 *  respectively.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array,
 *          dimension (LDA,N)
 *          On entry, the N-by-N coefficient matrix A.
 *          On exit, if iterative refinement has been successfully used
 *          (INFO.EQ.0 and ITER.GE.0, see description below), then A is
 *          unchanged, if double precision factorization has been used
 *          (INFO.EQ.0 and ITER.LT.0, see description below), then the
 *          array A contains the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          The pivot indices that define the permutation matrix P;
 *          row i of the matrix was interchanged with row IPIV(i).
 *          Corresponds either to the single precision factorization
 *          (if INFO.EQ.0 and ITER.GE.0) or the double precision
 *          factorization (if INFO.EQ.0 and ITER.LT.0).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The N-by-NRHS right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N,NRHS)
 *          This array is used to hold the residual vectors.
 *
 *  SWORK   (workspace) REAL array, dimension (N*(N+NRHS))
 *          This array is used to use the single precision matrix and the
 *          right-hand sides or solutions in single precision.
 *
 *  ITER    (output) INTEGER
 *          < 0: iterative refinement has failed, double precision
 *               factorization has been performed
 *               -1 : the routine fell back to full precision for
 *                    implementation- or machine-specific reasons
 *               -2 : narrowing the precision induced an overflow,
 *                    the routine fell back to full precision
 *               -3 : failure of SGETRF
 *               -31: stop the iterative refinement after the 30th
 *                    iterations
 *          > 0: iterative refinement has been sucessfully used.
 *               Returns the number of iterations
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, U(i,i) computed in DOUBLE PRECISION is
 *                exactly zero.  The factorization has been completed,
 *                but the factor U is exactly singular, so the solution
 *                could not be computed.
 *
 *  =========
 *
 *     .. Parameters ..
 **/
int C_DSGESV(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, double* x, int ldx, double* work, int* iter)
{
    int info;
    ::F_DSGESV(&n, &nrhs, a, &lda, ipiv, b, &ldb, x, &ldx, work, iter, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPCON estimates the reciprocal of the condition number (in the
 *  1-norm) of a real symmetric packed matrix A using the factorization
 *  A = U*D*U**T or A = L*D*L**T computed by DSPTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by DSPTRF, stored as a
 *          packed triangular matrix.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSPTRF.
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          The 1-norm of the original matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPCON(char uplo, int n, double* ap, int* ipiv, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DSPCON(&uplo, &n, ap, ipiv, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPEV computes all the eigenvalues and, optionally, eigenvectors of a
 *  real symmetric matrix A in packed storage.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, AP is overwritten by values generated during the
 *          reduction to tridiagonal form.  If UPLO = 'U', the diagonal
 *          and first superdiagonal of the tridiagonal matrix T overwrite
 *          the corresponding elements of A, and if UPLO = 'L', the
 *          diagonal and first subdiagonal of T overwrite the
 *          corresponding elements of A.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
 *          eigenvectors of the matrix A, with the i-th column of Z
 *          holding the eigenvector associated with W(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, the algorithm failed to converge; i
 *                off-diagonal elements of an intermediate tridiagonal
 *                form did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPEV(char jobz, char uplo, int n, double* ap, double* w, double* z, int ldz, double* work)
{
    int info;
    ::F_DSPEV(&jobz, &uplo, &n, ap, w, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPEVD computes all the eigenvalues and, optionally, eigenvectors
 *  of a real symmetric matrix A in packed storage. If eigenvectors are
 *  desired, it uses a divide and conquer algorithm.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, AP is overwritten by values generated during the
 *          reduction to tridiagonal form.  If UPLO = 'U', the diagonal
 *          and first superdiagonal of the tridiagonal matrix T overwrite
 *          the corresponding elements of A, and if UPLO = 'L', the
 *          diagonal and first subdiagonal of T overwrite the
 *          corresponding elements of A.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
 *          eigenvectors of the matrix A, with the i-th column of Z
 *          holding the eigenvector associated with W(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array,
 *                                         dimension (LWORK)
 *          On exit, if INFO = 0, WORK(1) returns the required LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If N <= 1,               LWORK must be at least 1.
 *          If JOBZ = 'N' and N > 1, LWORK must be at least 2*N.
 *          If JOBZ = 'V' and N > 1, LWORK must be at least
 *                                                 1 + 6*N + N**2.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the required sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the required LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If JOBZ  = 'N' or N <= 1, LIWORK must be at least 1.
 *          If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the required sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  if INFO = i, the algorithm failed to converge; i
 *                off-diagonal elements of an intermediate tridiagonal
 *                form did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPEVD(char jobz, char uplo, int n, double* ap, double* w, double* z, int ldz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSPEVD(&jobz, &uplo, &n, ap, w, z, &ldz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPEVX computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric matrix A in packed storage.  Eigenvalues/vectors
 *  can be selected by specifying either a range of values or a range of
 *  indices for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found;
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found;
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, AP is overwritten by values generated during the
 *          reduction to tridiagonal form.  If UPLO = 'U', the diagonal
 *          and first superdiagonal of the tridiagonal matrix T overwrite
 *          the corresponding elements of A, and if UPLO = 'L', the
 *          diagonal and first subdiagonal of T overwrite the
 *          corresponding elements of A.
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing AP to tridiagonal form.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *          See "Computing Small Singular Values of Bidiagonal Matrices
 *          with Guaranteed High Relative Accuracy," by Demmel and
 *          Kahan, LAPACK Working Note #3.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the selected eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          If an eigenvector fails to converge, then that column of Z
 *          contains the latest approximation to the eigenvector, and the
 *          index of the eigenvector is returned in IFAIL.
 *          If JOBZ = 'N', then Z is not referenced.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (8*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (N)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvectors that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, then i eigenvectors failed to converge.
 *                Their indices are stored in array IFAIL.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPEVX(char jobz, char range, char uplo, int n, double* ap, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int* iwork, int* ifail)
{
    int info;
    ::F_DSPEVX(&jobz, &range, &uplo, &n, ap, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPGST reduces a real symmetric-definite generalized eigenproblem
 *  to standard form, using packed storage.
 *
 *  If ITYPE = 1, the problem is A*x = lambda*B*x,
 *  and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
 *
 *  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 *  B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
 *
 *  B must have been previously factorized as U**T*U or L*L**T by DPPTRF.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
 *          = 2 or 3: compute U*A*U**T or L**T*A*L.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored and B is factored as
 *                  U**T*U;
 *          = 'L':  Lower triangle of A is stored and B is factored as
 *                  L*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, if INFO = 0, the transformed matrix, stored in the
 *          same format as A.
 *
 *  BP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The triangular factor from the Cholesky factorization of B,
 *          stored in the same format as A, as returned by DPPTRF.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPGST(int itype, char uplo, int n, double* ap, double* bp)
{
    int info;
    ::F_DSPGST(&itype, &uplo, &n, ap, bp, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPGV computes all the eigenvalues and, optionally, the eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
 *  Here A and B are assumed to be symmetric, stored in packed format,
 *  and B is also positive definite.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          Specifies the problem type to be solved:
 *          = 1:  A*x = (lambda)*B*x
 *          = 2:  A*B*x = (lambda)*x
 *          = 3:  B*A*x = (lambda)*x
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension
 *                            (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, the contents of AP are destroyed.
 *
 *  BP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          B, packed columnwise in a linear array.  The j-th column of B
 *          is stored in the array BP as follows:
 *          if UPLO = 'U', BP(i + (j-1)*j/2) = B(i,j) for 1<=i<=j;
 *          if UPLO = 'L', BP(i + (j-1)*(2*n-j)/2) = B(i,j) for j<=i<=n.
 *
 *          On exit, the triangular factor U or L from the Cholesky
 *          factorization B = U**T*U or B = L*L**T, in the same storage
 *          format as B.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of
 *          eigenvectors.  The eigenvectors are normalized as follows:
 *          if ITYPE = 1 or 2, Z**T*B*Z = I;
 *          if ITYPE = 3, Z**T*inv(B)*Z = I.
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  DPPTRF or DSPEV returned an error code:
 *             <= N:  if INFO = i, DSPEV failed to converge;
 *                    i off-diagonal elements of an intermediate
 *                    tridiagonal form did not converge to zero.
 *             > N:   if INFO = n + i, for 1 <= i <= n, then the leading
 *                    minor of order i of B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DSPGV(int itype, char jobz, char uplo, int n, double* ap, double* bp, double* w, double* z, int ldz, double* work)
{
    int info;
    ::F_DSPGV(&itype, &jobz, &uplo, &n, ap, bp, w, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPGVD computes all the eigenvalues, and optionally, the eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
 *  B are assumed to be symmetric, stored in packed format, and B is also
 *  positive definite.
 *  If eigenvectors are desired, it uses a divide and conquer algorithm.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          Specifies the problem type to be solved:
 *          = 1:  A*x = (lambda)*B*x
 *          = 2:  A*B*x = (lambda)*x
 *          = 3:  B*A*x = (lambda)*x
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, the contents of AP are destroyed.
 *
 *  BP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          B, packed columnwise in a linear array.  The j-th column of B
 *          is stored in the array BP as follows:
 *          if UPLO = 'U', BP(i + (j-1)*j/2) = B(i,j) for 1<=i<=j;
 *          if UPLO = 'L', BP(i + (j-1)*(2*n-j)/2) = B(i,j) for j<=i<=n.
 *
 *          On exit, the triangular factor U or L from the Cholesky
 *          factorization B = U**T*U or B = L*L**T, in the same storage
 *          format as B.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of
 *          eigenvectors.  The eigenvectors are normalized as follows:
 *          if ITYPE = 1 or 2, Z**T*B*Z = I;
 *          if ITYPE = 3, Z**T*inv(B)*Z = I.
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the required LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If N <= 1,               LWORK >= 1.
 *          If JOBZ = 'N' and N > 1, LWORK >= 2*N.
 *          If JOBZ = 'V' and N > 1, LWORK >= 1 + 6*N + 2*N**2.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the required sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the required LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If JOBZ  = 'N' or N <= 1, LIWORK >= 1.
 *          If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the required sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  DPPTRF or DSPEVD returned an error code:
 *             <= N:  if INFO = i, DSPEVD failed to converge;
 *                    i off-diagonal elements of an intermediate
 *                    tridiagonal form did not converge to zero;
 *             > N:   if INFO = N + i, for 1 <= i <= N, then the leading
 *                    minor of order i of B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPGVD(int itype, char jobz, char uplo, int n, double* ap, double* bp, double* w, double* z, int ldz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSPGVD(&itype, &jobz, &uplo, &n, ap, bp, w, z, &ldz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPGVX computes selected eigenvalues, and optionally, eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
 *  and B are assumed to be symmetric, stored in packed storage, and B
 *  is also positive definite.  Eigenvalues and eigenvectors can be
 *  selected by specifying either a range of values or a range of indices
 *  for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          Specifies the problem type to be solved:
 *          = 1:  A*x = (lambda)*B*x
 *          = 2:  A*B*x = (lambda)*x
 *          = 3:  B*A*x = (lambda)*x
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A and B are stored;
 *          = 'L':  Lower triangle of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix pencil (A,B).  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, the contents of AP are destroyed.
 *
 *  BP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          B, packed columnwise in a linear array.  The j-th column of B
 *          is stored in the array BP as follows:
 *          if UPLO = 'U', BP(i + (j-1)*j/2) = B(i,j) for 1<=i<=j;
 *          if UPLO = 'L', BP(i + (j-1)*(2*n-j)/2) = B(i,j) for j<=i<=n.
 *
 *          On exit, the triangular factor U or L from the Cholesky
 *          factorization B = U**T*U or B = L*L**T, in the same storage
 *          format as B.
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing A to tridiagonal form.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          On normal exit, the first M elements contain the selected
 *          eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))
 *          If JOBZ = 'N', then Z is not referenced.
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          The eigenvectors are normalized as follows:
 *          if ITYPE = 1 or 2, Z**T*B*Z = I;
 *          if ITYPE = 3, Z**T*inv(B)*Z = I.
 *
 *          If an eigenvector fails to converge, then that column of Z
 *          contains the latest approximation to the eigenvector, and the
 *          index of the eigenvector is returned in IFAIL.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (8*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (N)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvectors that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  DPPTRF or DSPEVX returned an error code:
 *             <= N:  if INFO = i, DSPEVX failed to converge;
 *                    i eigenvectors failed to converge.  Their indices
 *                    are stored in array IFAIL.
 *             > N:   if INFO = N + i, for 1 <= i <= N, then the leading
 *                    minor of order i of B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
 *
 * =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DSPGVX(int itype, char jobz, char range, char uplo, int n, double* ap, double* bp, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int* iwork, int* ifail)
{
    int info;
    ::F_DSPGVX(&itype, &jobz, &range, &uplo, &n, ap, bp, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is symmetric indefinite
 *  and packed, and provides error bounds and backward error estimates
 *  for the solution.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangle of the symmetric matrix A, packed
 *          columnwise in a linear array.  The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *  AFP     (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The factored form of the matrix A.  AFP contains the block
 *          diagonal matrix D and the multipliers used to obtain the
 *          factor U or L from the factorization A = U*D*U**T or
 *          A = L*D*L**T as computed by DSPTRF, stored as a packed
 *          triangular matrix.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSPTRF.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DSPTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPRFS(char uplo, int n, int nrhs, double* ap, double* afp, int* ipiv, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DSPRFS(&uplo, &n, &nrhs, ap, afp, ipiv, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPSV computes the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric matrix stored in packed format and X
 *  and B are N-by-NRHS matrices.
 *
 *  The diagonal pivoting method is used to factor A as
 *     A = U * D * U**T,  if UPLO = 'U', or
 *     A = L * D * L**T,  if UPLO = 'L',
 *  where U (or L) is a product of permutation and unit upper (lower)
 *  triangular matrices, D is symmetric and block diagonal with 1-by-1
 *  and 2-by-2 diagonal blocks.  The factored form of A is then used to
 *  solve the system of equations A * X = B.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *          See below for further details.
 *
 *          On exit, the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L from the factorization
 *          A = U*D*U**T or A = L*D*L**T as computed by DSPTRF, stored as
 *          a packed triangular matrix in the same storage format as A.
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D, as
 *          determined by DSPTRF.  If IPIV(k) > 0, then rows and columns
 *          k and IPIV(k) were interchanged, and D(k,k) is a 1-by-1
 *          diagonal block.  If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0,
 *          then rows and columns k-1 and -IPIV(k) were interchanged and
 *          D(k-1:k,k-1:k) is a 2-by-2 diagonal block.  If UPLO = 'L' and
 *          IPIV(k) = IPIV(k+1) < 0, then rows and columns k+1 and
 *          -IPIV(k) were interchanged and D(k:k+1,k:k+1) is a 2-by-2
 *          diagonal block.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, D(i,i) is exactly zero.  The factorization
 *                has been completed, but the block diagonal matrix D is
 *                exactly singular, so the solution could not be
 *                computed.
 *
 *  Further Details
 *  ===============
 *
 *  The packed storage scheme is illustrated by the following example
 *  when N = 4, UPLO = 'U':
 *
 *  Two-dimensional storage of the symmetric matrix A:
 *
 *     a11 a12 a13 a14
 *         a22 a23 a24
 *             a33 a34     (aij = aji)
 *                 a44
 *
 *  Packed storage of the upper triangle of A:
 *
 *  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
 *
 *  =====================================================================
 *
 *     .. External Functions ..
 **/
int C_DSPSV(char uplo, int n, int nrhs, double* ap, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DSPSV(&uplo, &n, &nrhs, ap, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPSVX uses the diagonal pivoting factorization A = U*D*U**T or
 *  A = L*D*L**T to compute the solution to a real system of linear
 *  equations A * X = B, where A is an N-by-N symmetric matrix stored
 *  in packed format and X and B are N-by-NRHS matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'N', the diagonal pivoting method is used to factor A as
 *        A = U * D * U**T,  if UPLO = 'U', or
 *        A = L * D * L**T,  if UPLO = 'L',
 *     where U (or L) is a product of permutation and unit upper (lower)
 *     triangular matrices and D is symmetric and block diagonal with
 *     1-by-1 and 2-by-2 diagonal blocks.
 *
 *  2. If some D(i,i)=0, so that D is exactly singular, then the routine
 *     returns with INFO = i. Otherwise, the factored form of A is used
 *     to estimate the condition number of the matrix A.  If the
 *     reciprocal of the condition number is less than machine precision,
 *  C++ Return value: INFO    (output) INTEGER
 *     to solve for X and compute error bounds as described below.
 *
 *  3. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  4. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of A has been
 *          supplied on entry.
 *          = 'F':  On entry, AFP and IPIV contain the factored form of
 *                  A.  AP, AFP and IPIV will not be modified.
 *          = 'N':  The matrix A will be copied to AFP and factored.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangle of the symmetric matrix A, packed
 *          columnwise in a linear array.  The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *          See below for further details.
 *
 *  AFP     (input or output) DOUBLE PRECISION array, dimension
 *                            (N*(N+1)/2)
 *          If FACT = 'F', then AFP is an input argument and on entry
 *          contains the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L from the factorization
 *          A = U*D*U**T or A = L*D*L**T as computed by DSPTRF, stored as
 *          a packed triangular matrix in the same storage format as A.
 *
 *          If FACT = 'N', then AFP is an output argument and on exit
 *          contains the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L from the factorization
 *          A = U*D*U**T or A = L*D*L**T as computed by DSPTRF, stored as
 *          a packed triangular matrix in the same storage format as A.
 *
 *  IPIV    (input or output) INTEGER array, dimension (N)
 *          If FACT = 'F', then IPIV is an input argument and on entry
 *          contains details of the interchanges and the block structure
 *          of D, as determined by DSPTRF.
 *          If IPIV(k) > 0, then rows and columns k and IPIV(k) were
 *          interchanged and D(k,k) is a 1-by-1 diagonal block.
 *          If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and
 *          columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
 *          is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =
 *          IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
 *          interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
 *
 *          If FACT = 'N', then IPIV is an output argument and on exit
 *          contains details of the interchanges and the block structure
 *          of D, as determined by DSPTRF.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The N-by-NRHS right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A.  If RCOND is less than the machine precision (in
 *          particular, if RCOND = 0), the matrix is singular to working
 *          precision.  This condition is indicated by a return code of
 *  C++ Return value: INFO    (output) INTEGER
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, and i is
 *                <= N:  D(i,i) is exactly zero.  The factorization
 *                       has been completed but the factor D is exactly
 *                       singular, so the solution and error bounds could
 *                       not be computed. RCOND = 0 is returned.
 *                = N+1: D is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  Further Details
 *  ===============
 *
 *  The packed storage scheme is illustrated by the following example
 *  when N = 4, UPLO = 'U':
 *
 *  Two-dimensional storage of the symmetric matrix A:
 *
 *     a11 a12 a13 a14
 *         a22 a23 a24
 *             a33 a34     (aij = aji)
 *                 a44
 *
 *  Packed storage of the upper triangle of A:
 *
 *  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPSVX(char fact, char uplo, int n, int nrhs, double* ap, double* afp, int* ipiv, double* b, int ldb, double* x, int ldx, double* rcond)
{
    int info;
    ::F_DSPSVX(&fact, &uplo, &n, &nrhs, ap, afp, ipiv, b, &ldb, x, &ldx, rcond, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPTRD reduces a real symmetric matrix A stored in packed form to
 *  symmetric tridiagonal form T by an orthogonal similarity
 *  transformation: Q**T * A * Q = T.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *          On exit, if UPLO = 'U', the diagonal and first superdiagonal
 *          of A are overwritten by the corresponding elements of the
 *          tridiagonal matrix T, and the elements above the first
 *          superdiagonal, with the array TAU, represent the orthogonal
 *          matrix Q as a product of elementary reflectors; if UPLO
 *          = 'L', the diagonal and first subdiagonal of A are over-
 *          written by the corresponding elements of the tridiagonal
 *          matrix T, and the elements below the first subdiagonal, with
 *          the array TAU, represent the orthogonal matrix Q as a product
 *          of elementary reflectors. See Further Details.
 *
 *  D       (output) DOUBLE PRECISION array, dimension (N)
 *          The diagonal elements of the tridiagonal matrix T:
 *          D(i) = A(i,i).
 *
 *  E       (output) DOUBLE PRECISION array, dimension (N-1)
 *          The off-diagonal elements of the tridiagonal matrix T:
 *          E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (N-1)
 *          The scalar factors of the elementary reflectors (see Further
 *          Details).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  If UPLO = 'U', the matrix Q is represented as a product of elementary
 *  reflectors
 *
 *     Q = H(n-1) . . . H(2) H(1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in AP,
 *  overwriting A(1:i-1,i+1), and tau is stored in TAU(i).
 *
 *  If UPLO = 'L', the matrix Q is represented as a product of elementary
 *  reflectors
 *
 *     Q = H(1) H(2) . . . H(n-1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in AP,
 *  overwriting A(i+2:n,i), and tau is stored in TAU(i).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPTRD(char uplo, int n, double* ap, double* d, double* e, double* tau)
{
    int info;
    ::F_DSPTRD(&uplo, &n, ap, d, e, tau, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPTRF computes the factorization of a real symmetric matrix A stored
 *  in packed format using the Bunch-Kaufman diagonal pivoting method:
 *
 *     A = U*D*U**T  or  A = L*D*L**T
 *
 *  where U (or L) is a product of permutation and unit upper (lower)
 *  triangular matrices, and D is symmetric and block diagonal with
 *  1-by-1 and 2-by-2 diagonal blocks.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangle of the symmetric matrix
 *          A, packed columnwise in a linear array.  The j-th column of A
 *          is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *
 *          On exit, the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L, stored as a packed triangular
 *          matrix overwriting A (see below for further details).
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D.
 *          If IPIV(k) > 0, then rows and columns k and IPIV(k) were
 *          interchanged and D(k,k) is a 1-by-1 diagonal block.
 *          If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and
 *          columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
 *          is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =
 *          IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
 *          interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, D(i,i) is exactly zero.  The factorization
 *               has been completed, but the block diagonal matrix D is
 *               exactly singular, and division by zero will occur if it
 *               is used to solve a system of equations.
 *
 *  Further Details
 *  ===============
 *
 *  5-96 - Based on modifications by J. Lewis, Boeing Computer Services
 *         Company
 *
 *  If UPLO = 'U', then A = U*D*U', where
 *     U = P(n)*U(n)* ... *P(k)U(k)* ...,
 *  i.e., U is a product of terms P(k)*U(k), where k decreases from n to
 *  1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
 *  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
 *  defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
 *  that if the diagonal block D(k) is of order s (s = 1 or 2), then
 *
 *             (   I    v    0   )   k-s
 *     U(k) =  (   0    I    0   )   s
 *             (   0    0    I   )   n-k
 *                k-s   s   n-k
 *
 *  If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
 *  If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
 *  and A(k,k), and v overwrites A(1:k-2,k-1:k).
 *
 *  If UPLO = 'L', then A = L*D*L', where
 *     L = P(1)*L(1)* ... *P(k)*L(k)* ...,
 *  i.e., L is a product of terms P(k)*L(k), where k increases from 1 to
 *  n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
 *  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
 *  defined by IPIV(k), and L(k) is a unit lower triangular matrix, such
 *  that if the diagonal block D(k) is of order s (s = 1 or 2), then
 *
 *             (   I    0     0   )  k-1
 *     L(k) =  (   0    I     0   )  s
 *             (   0    v     I   )  n-k-s+1
 *                k-1   s  n-k-s+1
 *
 *  If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
 *  If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
 *  and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPTRF(char uplo, int n, double* ap, int* ipiv)
{
    int info;
    ::F_DSPTRF(&uplo, &n, ap, ipiv, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPTRI computes the inverse of a real symmetric indefinite matrix
 *  A in packed storage using the factorization A = U*D*U**T or
 *  A = L*D*L**T computed by DSPTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the block diagonal matrix D and the multipliers
 *          used to obtain the factor U or L as computed by DSPTRF,
 *          stored as a packed triangular matrix.
 *
 *          On exit, if INFO = 0, the (symmetric) inverse of the original
 *          matrix, stored as a packed triangular matrix. The j-th column
 *          of inv(A) is stored in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = inv(A)(i,j) for 1<=i<=j;
 *          if UPLO = 'L',
 *             AP(i + (j-1)*(2n-j)/2) = inv(A)(i,j) for j<=i<=n.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSPTRF.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, D(i,i) = 0; the matrix is singular and its
 *               inverse could not be computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPTRI(char uplo, int n, double* ap, int* ipiv, double* work)
{
    int info;
    ::F_DSPTRI(&uplo, &n, ap, ipiv, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSPTRS solves a system of linear equations A*X = B with a real
 *  symmetric matrix A stored in packed format using the factorization
 *  A = U*D*U**T or A = L*D*L**T computed by DSPTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by DSPTRF, stored as a
 *          packed triangular matrix.
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSPTRF.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSPTRS(char uplo, int n, int nrhs, double* ap, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DSPTRS(&uplo, &n, &nrhs, ap, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEBZ computes the eigenvalues of a symmetric tridiagonal
 *  matrix T.  The user may ask for all eigenvalues, all eigenvalues
 *  in the half-open interval (VL, VU], or the IL-th through IU-th
 *  eigenvalues.
 *
 *  To avoid overflow, the matrix must be scaled so that its
 *  largest element is no greater than overflow**(1/2) *
 *  underflow**(1/4) in absolute value, and for greatest
 *  accuracy, it should not be much smaller than that.
 *
 *  See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
 *  Matrix", Report CS41, Computer Science Dept., Stanford
 *  University, July 21, 1966.
 *
 *  Arguments
 *  =========
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': ("All")   all eigenvalues will be found.
 *          = 'V': ("Value") all eigenvalues in the half-open interval
 *                           (VL, VU] will be found.
 *          = 'I': ("Index") the IL-th through IU-th eigenvalues (of the
 *                           entire matrix) will be found.
 *
 *  ORDER   (input) CHARACTER*1
 *          = 'B': ("By Block") the eigenvalues will be grouped by
 *                              split-off block (see IBLOCK, ISPLIT) and
 *                              ordered from smallest to largest within
 *                              the block.
 *          = 'E': ("Entire matrix")
 *                              the eigenvalues for the entire matrix
 *                              will be ordered from smallest to
 *                              largest.
 *
 *  N       (input) INTEGER
 *          The order of the tridiagonal matrix T.  N >= 0.
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues.  Eigenvalues less than or equal
 *          to VL, or greater than VU, will not be returned.  VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute tolerance for the eigenvalues.  An eigenvalue
 *          (or cluster) is considered to be located if it has been
 *          determined to lie in an interval whose width is ABSTOL or
 *          less.  If ABSTOL is less than or equal to zero, then ULP*|T|
 *          will be used, where |T| means the 1-norm of T.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the tridiagonal matrix T.
 *
 *  E       (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) off-diagonal elements of the tridiagonal matrix T.
 *
 *  M       (output) INTEGER
 *          The actual number of eigenvalues found. 0 <= M <= N.
 *          (See also the description of INFO=2,3.)
 *
 *  NSPLIT  (output) INTEGER
 *          The number of diagonal blocks in the matrix T.
 *          1 <= NSPLIT <= N.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, the first M elements of W will contain the
 *          eigenvalues.  (DSTEBZ may use the remaining N-M elements as
 *          workspace.)
 *
 *  IBLOCK  (output) INTEGER array, dimension (N)
 *          At each row/column j where E(j) is zero or small, the
 *          matrix T is considered to split into a block diagonal
 *          matrix.  On exit, if INFO = 0, IBLOCK(i) specifies to which
 *          block (from 1 to the number of blocks) the eigenvalue W(i)
 *          belongs.  (DSTEBZ may use the remaining N-M elements as
 *          workspace.)
 *
 *  ISPLIT  (output) INTEGER array, dimension (N)
 *          The splitting points, at which T breaks up into submatrices.
 *          The first submatrix consists of rows/columns 1 to ISPLIT(1),
 *          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2),
 *          etc., and the NSPLIT-th consists of rows/columns
 *          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N.
 *          (Only the first NSPLIT elements will actually be used, but
 *          since the user cannot know a priori what value NSPLIT will
 *          have, N words must be reserved for ISPLIT.)
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (4*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (3*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  some or all of the eigenvalues failed to converge or
 *                were not computed:
 *                =1 or 3: Bisection failed to converge for some
 *                        eigenvalues; these eigenvalues are flagged by a
 *                        negative block number.  The effect is that the
 *                        eigenvalues may not be as accurate as the
 *                        absolute and relative tolerances.  This is
 *                        generally caused by unexpectedly inaccurate
 *                        arithmetic.
 *                =2 or 3: RANGE='I' only: Not all of the eigenvalues
 *                        IL:IU were found.
 *                        Effect: M < IU+1-IL
 *                        Cause:  non-monotonic arithmetic, causing the
 *                                Sturm sequence to be non-monotonic.
 *                        Cure:   recalculate, using RANGE='A', and pick
 *                                out eigenvalues IL:IU.  In some cases,
 *                                increasing the PARAMETER "FUDGE" may
 *                                make things work.
 *                = 4:    RANGE='I', and the Gershgorin interval
 *                        initially used was too small.  No eigenvalues
 *                        were computed.
 *                        Probable cause: your machine has sloppy
 *                                        floating-point arithmetic.
 *                        Cure: Increase the PARAMETER "FUDGE",
 *                              recompile, and try again.
 *
 *  Internal Parameters
 *  ===================
 *
 *  RELFAC  DOUBLE PRECISION, default = 2.0e0
 *          The relative tolerance.  An interval (a,b] lies within
 *          "relative tolerance" if  b-a < RELFAC*ulp*max(|a|,|b|),
 *          where "ulp" is the machine precision (distance from 1 to
 *          the next larger floating point number.)
 *
 *  FUDGE   DOUBLE PRECISION, default = 2
 *          A "fudge factor" to widen the Gershgorin intervals.  Ideally,
 *          a value of 1 should work, but on machines with sloppy
 *          arithmetic, this needs to be larger.  The default for
 *          publicly released versions should be large enough to handle
 *          the worst machine around.  Note that this has no effect
 *          on accuracy of the solution.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEBZ(char range, char order, int n, double vl, double vu, int il, int iu, double abstol, double* d, double* e, int* m, int* nsplit, double* w, int* iblock, int* isplit, double* work, int* iwork)
{
    int info;
    ::F_DSTEBZ(&range, &order, &n, &vl, &vu, &il, &iu, &abstol, d, e, m, nsplit, w, iblock, isplit, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEDC computes all eigenvalues and, optionally, eigenvectors of a
 *  symmetric tridiagonal matrix using the divide and conquer method.
 *  The eigenvectors of a full or band real symmetric matrix can also be
 *  found if DSYTRD or DSPTRD or DSBTRD has been used to reduce this
 *  matrix to tridiagonal form.
 *
 *  This code makes very mild assumptions about floating point
 *  arithmetic. It will work on machines with a guard digit in
 *  add/subtract, or on those binary machines without guard digits
 *  which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
 *  It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.  See DLAED3 for details.
 *
 *  Arguments
 *  =========
 *
 *  COMPZ   (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only.
 *          = 'I':  Compute eigenvectors of tridiagonal matrix also.
 *          = 'V':  Compute eigenvectors of original dense symmetric
 *                  matrix also.  On entry, Z contains the orthogonal
 *                  matrix used to reduce the original matrix to
 *                  tridiagonal form.
 *
 *  N       (input) INTEGER
 *          The dimension of the symmetric tridiagonal matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the diagonal elements of the tridiagonal matrix.
 *          On exit, if INFO = 0, the eigenvalues in ascending order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the subdiagonal elements of the tridiagonal matrix.
 *          On exit, E has been destroyed.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ,N)
 *          On entry, if COMPZ = 'V', then Z contains the orthogonal
 *          matrix used in the reduction to tridiagonal form.
 *          On exit, if INFO = 0, then if COMPZ = 'V', Z contains the
 *          orthonormal eigenvectors of the original symmetric matrix,
 *          and if COMPZ = 'I', Z contains the orthonormal eigenvectors
 *          of the symmetric tridiagonal matrix.
 *          If  COMPZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1.
 *          If eigenvectors are desired, then LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array,
 *                                         dimension (LWORK)
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If COMPZ = 'N' or N <= 1 then LWORK must be at least 1.
 *          If COMPZ = 'V' and N > 1 then LWORK must be at least
 *                         ( 1 + 3*N + 2*N*lg N + 3*N**2 ),
 *                         where lg( N ) = smallest integer k such
 *                         that 2**k >= N.
 *          If COMPZ = 'I' and N > 1 then LWORK must be at least
 *                         ( 1 + 4*N + N**2 ).
 *          Note that for COMPZ = 'I' or 'V', then if N is less than or
 *          equal to the minimum divide size, usually 25, then LWORK need
 *          only be max(1,2*(N-1)).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If COMPZ = 'N' or N <= 1 then LIWORK must be at least 1.
 *          If COMPZ = 'V' and N > 1 then LIWORK must be at least
 *                         ( 6 + 6*N + 5*N*lg N ).
 *          If COMPZ = 'I' and N > 1 then LIWORK must be at least
 *                         ( 3 + 5*N ).
 *          Note that for COMPZ = 'I' or 'V', then if N is less than or
 *          equal to the minimum divide size, usually 25, then LIWORK
 *          need only be 1.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal size of the IWORK array,
 *          returns this value as the first entry of the IWORK array, and
 *          no error message related to LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  The algorithm failed to compute an eigenvalue while
 *                working on the submatrix lying in rows and columns
 *  C++ Return value: INFO    (output) INTEGER
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Jeff Rutter, Computer Science Division, University of California
 *     at Berkeley, USA
 *  Modified by Francoise Tisseur, University of Tennessee.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEDC(char compz, int n, double* d, double* e, double* z, int ldz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSTEDC(&compz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEGR computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric tridiagonal matrix T. Any such unreduced matrix has
 *  a well defined set of pairwise different real eigenvalues, the corresponding
 *  real eigenvectors are pairwise orthogonal.
 *
 *  The spectrum may be computed either completely or partially by specifying
 *  either an interval (VL,VU] or a range of indices IL:IU for the desired
 *  eigenvalues.
 *
 *  DSTEGR is a compatability wrapper around the improved DSTEMR routine.
 *  See DSTEMR for further details.
 *
 *  One important change is that the ABSTOL parameter no longer provides any
 *  benefit and hence is no longer used.
 *
 *  Note : DSTEGR and DSTEMR work only on machines which follow
 *  IEEE-754 floating-point standard in their handling of infinities and
 *  NaNs.  Normal execution may create these exceptiona values and hence
 *  may abort due to a floating point exception in environments which
 *  do not conform to the IEEE-754 standard.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the N diagonal elements of the tridiagonal matrix
 *          T. On exit, D is overwritten.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the (N-1) subdiagonal elements of the tridiagonal
 *          matrix T in elements 1 to N-1 of E. E(N) need not be set on
 *          input, but is used internally as workspace.
 *          On exit, E is overwritten.
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          Unused.  Was the absolute error tolerance for the
 *          eigenvalues/eigenvectors in previous versions.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          The first M elements contain the selected eigenvalues in
 *          ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M) )
 *          If JOBZ = 'V', and if INFO = 0, then the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix T
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *          Supplying N columns is always safe.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', then LDZ >= max(1,N).
 *
 *  ISUPPZ  (output) INTEGER ARRAY, dimension ( 2*max(1,M) )
 *          The support of the eigenvectors in Z, i.e., the indices
 *          indicating the nonzero elements in Z. The i-th computed eigenvector
 *          is nonzero only in elements ISUPPZ( 2*i-1 ) through
 *          ISUPPZ( 2*i ). This is relevant in the case when the matrix
 *          is split. ISUPPZ is only accessed when JOBZ is 'V' and N > 0.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
 *          On exit, if INFO = 0, WORK(1) returns the optimal
 *          (and minimal) LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,18*N)
 *          if JOBZ = 'V', and LWORK >= max(1,12*N) if JOBZ = 'N'.
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (LIWORK)
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.  LIWORK >= max(1,10*N)
 *          if the eigenvectors are desired, and LIWORK >= max(1,8*N)
 *          if only the eigenvalues are to be computed.
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal size of the IWORK array,
 *          returns this value as the first entry of the IWORK array, and
 *          no error message related to LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          On exit, INFO
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = 1X, internal error in DLARRE,
 *                if INFO = 2X, internal error in DLARRV.
 *                Here, the digit X = ABS( IINFO ) < 10, where IINFO is
 *                the nonzero error code returned by DLARRE or
 *                DLARRV, respectively.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Inderjit Dhillon, IBM Almaden, USA
 *     Osni Marques, LBNL/NERSC, USA
 *     Christof Voemel, LBNL/NERSC, USA
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DSTEGR(char jobz, char range, int n, double* d, double* e, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, int* isuppz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSTEGR(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEIN computes the eigenvectors of a real symmetric tridiagonal
 *  matrix T corresponding to specified eigenvalues, using inverse
 *  iteration.
 *
 *  The maximum number of iterations allowed for each eigenvector is
 *  specified by an internal parameter MAXITS (currently set to 5).
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input) DOUBLE PRECISION array, dimension (N)
 *          The n diagonal elements of the tridiagonal matrix T.
 *
 *  E       (input) DOUBLE PRECISION array, dimension (N-1)
 *          The (n-1) subdiagonal elements of the tridiagonal matrix
 *          T, in elements 1 to N-1.
 *
 *  M       (input) INTEGER
 *          The number of eigenvectors to be found.  0 <= M <= N.
 *
 *  W       (input) DOUBLE PRECISION array, dimension (N)
 *          The first M elements of W contain the eigenvalues for
 *          which eigenvectors are to be computed.  The eigenvalues
 *          should be grouped by split-off block and ordered from
 *          smallest to largest within the block.  ( The output array
 *          W from DSTEBZ with ORDER = 'B' is expected here. )
 *
 *  IBLOCK  (input) INTEGER array, dimension (N)
 *          The submatrix indices associated with the corresponding
 *          eigenvalues in W; IBLOCK(i)=1 if eigenvalue W(i) belongs to
 *          the first submatrix from the top, =2 if W(i) belongs to
 *          the second submatrix, etc.  ( The output array IBLOCK
 *          from DSTEBZ is expected here. )
 *
 *  ISPLIT  (input) INTEGER array, dimension (N)
 *          The splitting points, at which T breaks up into submatrices.
 *          The first submatrix consists of rows/columns 1 to
 *          ISPLIT( 1 ), the second of rows/columns ISPLIT( 1 )+1
 *          through ISPLIT( 2 ), etc.
 *          ( The output array ISPLIT from DSTEBZ is expected here. )
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, M)
 *          The computed eigenvectors.  The eigenvector associated
 *          with the eigenvalue W(i) is stored in the i-th column of
 *          Z.  Any vector which fails to converge is set to its current
 *          iterate after MAXITS iterations.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (5*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  IFAIL   (output) INTEGER array, dimension (M)
 *          On normal exit, all elements of IFAIL are zero.
 *          If one or more eigenvectors fail to converge after
 *          MAXITS iterations, then their indices are stored in
 *          array IFAIL.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit.
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, then i eigenvectors failed to converge
 *               in MAXITS iterations.  Their indices are stored in
 *               array IFAIL.
 *
 *  Internal Parameters
 *  ===================
 *
 *  MAXITS  INTEGER, default = 5
 *          The maximum number of iterations performed.
 *
 *  EXTRA   INTEGER, default = 2
 *          The number of iterations performed after norm growth
 *          criterion is satisfied, should be at least 1.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEIN(int n, double* d, double* e, int m, double* w, int* iblock, int* isplit, double* z, int ldz, double* work, int* iwork, int* ifail)
{
    int info;
    ::F_DSTEIN(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEQR computes all eigenvalues and, optionally, eigenvectors of a
 *  symmetric tridiagonal matrix using the implicit QL or QR method.
 *  The eigenvectors of a full or band symmetric matrix can also be found
 *  if DSYTRD or DSPTRD or DSBTRD has been used to reduce this matrix to
 *  tridiagonal form.
 *
 *  Arguments
 *  =========
 *
 *  COMPZ   (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only.
 *          = 'V':  Compute eigenvalues and eigenvectors of the original
 *                  symmetric matrix.  On entry, Z must contain the
 *                  orthogonal matrix used to reduce the original matrix
 *                  to tridiagonal form.
 *          = 'I':  Compute eigenvalues and eigenvectors of the
 *                  tridiagonal matrix.  Z is initialized to the identity
 *                  matrix.
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the diagonal elements of the tridiagonal matrix.
 *          On exit, if INFO = 0, the eigenvalues in ascending order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix.
 *          On exit, E has been destroyed.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          On entry, if  COMPZ = 'V', then Z contains the orthogonal
 *          matrix used in the reduction to tridiagonal form.
 *          On exit, if INFO = 0, then if  COMPZ = 'V', Z contains the
 *          orthonormal eigenvectors of the original symmetric matrix,
 *          and if COMPZ = 'I', Z contains the orthonormal eigenvectors
 *          of the symmetric tridiagonal matrix.
 *          If COMPZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          eigenvectors are desired, then  LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (max(1,2*N-2))
 *          If COMPZ = 'N', then WORK is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  the algorithm has failed to find all the eigenvalues in
 *                a total of 30*N iterations; if INFO = i, then i
 *                elements of E have not converged to zero; on exit, D
 *                and E contain the elements of a symmetric tridiagonal
 *                matrix which is orthogonally similar to the original
 *                matrix.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEQR(char compz, int n, double* d, double* e, double* z, int ldz, double* work)
{
    int info;
    ::F_DSTEQR(&compz, &n, d, e, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTERF computes all eigenvalues of a symmetric tridiagonal matrix
 *  using the Pal-Walker-Kahan variant of the QL or QR algorithm.
 *
 *  Arguments
 *  =========
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix.
 *          On exit, if INFO = 0, the eigenvalues in ascending order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix.
 *          On exit, E has been destroyed.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  the algorithm failed to find all of the eigenvalues in
 *                a total of 30*N iterations; if INFO = i, then i
 *                elements of E have not converged to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTERF(int n, double* d, double* e)
{
    int info;
    ::F_DSTERF(&n, d, e, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEV computes all eigenvalues and, optionally, eigenvectors of a
 *  real symmetric tridiagonal matrix A.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix
 *          A.
 *          On exit, if INFO = 0, the eigenvalues in ascending order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix A, stored in elements 1 to N-1 of E.
 *          On exit, the contents of E are destroyed.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
 *          eigenvectors of the matrix A, with the i-th column of Z
 *          holding the eigenvector associated with D(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (max(1,2*N-2))
 *          If JOBZ = 'N', WORK is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the algorithm failed to converge; i
 *                off-diagonal elements of E did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEV(char jobz, int n, double* d, double* e, double* z, int ldz, double* work)
{
    int info;
    ::F_DSTEV(&jobz, &n, d, e, z, &ldz, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEVD computes all eigenvalues and, optionally, eigenvectors of a
 *  real symmetric tridiagonal matrix. If eigenvectors are desired, it
 *  uses a divide and conquer algorithm.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix
 *          A.
 *          On exit, if INFO = 0, the eigenvalues in ascending order.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (N-1)
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix A, stored in elements 1 to N-1 of E.
 *          On exit, the contents of E are destroyed.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, N)
 *          If JOBZ = 'V', then if INFO = 0, Z contains the orthonormal
 *          eigenvectors of the matrix A, with the i-th column of Z
 *          holding the eigenvector associated with D(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array,
 *                                         dimension (LWORK)
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If JOBZ  = 'N' or N <= 1 then LWORK must be at least 1.
 *          If JOBZ  = 'V' and N > 1 then LWORK must be at least
 *                         ( 1 + 4*N + N**2 ).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If JOBZ  = 'N' or N <= 1 then LIWORK must be at least 1.
 *          If JOBZ  = 'V' and N > 1 then LIWORK must be at least 3+5*N.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the algorithm failed to converge; i
 *                off-diagonal elements of E did not converge to zero.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEVD(char jobz, int n, double* d, double* e, double* z, int ldz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSTEVD(&jobz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEVR computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric tridiagonal matrix T.  Eigenvalues and
 *  eigenvectors can be selected by specifying either a range of values
 *  or a range of indices for the desired eigenvalues.
 *
 *  Whenever possible, DSTEVR calls DSTEMR to compute the
 *  eigenspectrum using Relatively Robust Representations.  DSTEMR
 *  computes eigenvalues by the dqds algorithm, while orthogonal
 *  eigenvectors are computed from various "good" L D L^T representations
 *  (also known as Relatively Robust Representations). Gram-Schmidt
 *  orthogonalization is avoided as far as possible. More specifically,
 *  the various steps of the algorithm are as follows. For the i-th
 *  unreduced block of T,
 *     (a) Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T
 *          is a relatively robust representation,
 *     (b) Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high
 *         relative accuracy by the dqds algorithm,
 *     (c) If there is a cluster of close eigenvalues, "choose" sigma_i
 *         close to the cluster, and go to step (a),
 *     (d) Given the approximate eigenvalue lambda_j of L_i D_i L_i^T,
 *         compute the corresponding eigenvector by forming a
 *         rank-revealing twisted factorization.
 *  The desired accuracy of the output can be specified by the input
 *  parameter ABSTOL.
 *
 *  For more details, see "A new O(n^2) algorithm for the symmetric
 *  tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon,
 *  Computer Science Division Technical Report No. UCB//CSD-97-971,
 *  UC Berkeley, May 1997.
 *
 *
 *  Note 1 : DSTEVR calls DSTEMR when the full spectrum is requested
 *  on machines which conform to the ieee-754 floating point standard.
 *  DSTEVR calls DSTEBZ and DSTEIN on non-ieee machines and
 *  when partial spectrum requests are made.
 *
 *  Normal execution of DSTEMR may create NaNs and infinities and
 *  hence may abort due to a floating point exception in environments
 *  which do not handle NaNs and infinities in the ieee standard default
 *  manner.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 ********** For RANGE = 'V' or 'I' and IU - IL < N - 1, DSTEBZ and
 ********** DSTEIN are called
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix
 *          A.
 *          On exit, D may be multiplied by a constant factor chosen
 *          to avoid over/underflow in computing the eigenvalues.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (max(1,N-1))
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix A in elements 1 to N-1 of E.
 *          On exit, E may be multiplied by a constant factor chosen
 *          to avoid over/underflow in computing the eigenvalues.
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing A to tridiagonal form.
 *
 *          See "Computing Small Singular Values of Bidiagonal Matrices
 *          with Guaranteed High Relative Accuracy," by Demmel and
 *          Kahan, LAPACK Working Note #3.
 *
 *          If high relative accuracy is important, set ABSTOL to
 *          DLAMCH( 'Safe minimum' ).  Doing so will guarantee that
 *          eigenvalues are computed to high relative accuracy when
 *          possible in future releases.  The current code does not
 *          make any guarantees about high relative accuracy, but
 *          future releases will. See J. Barlow and J. Demmel,
 *          "Computing Accurate Eigensystems of Scaled Diagonally
 *          Dominant Matrices", LAPACK Working Note #7, for a discussion
 *          of which matrices define their eigenvalues to high relative
 *          accuracy.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          The first M elements contain the selected eigenvalues in
 *          ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M) )
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  ISUPPZ  (output) INTEGER array, dimension ( 2*max(1,M) )
 *          The support of the eigenvectors in Z, i.e., the indices
 *          indicating the nonzero elements in Z. The i-th eigenvector
 *          is nonzero only in elements ISUPPZ( 2*i-1 ) through
 *          ISUPPZ( 2*i ).
 ********** Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal (and
 *          minimal) LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,20*N).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal (and
 *          minimal) LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.  LIWORK >= max(1,10*N).
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  Internal error
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Inderjit Dhillon, IBM Almaden, USA
 *     Osni Marques, LBNL/NERSC, USA
 *     Ken Stanley, Computer Science Division, University of
 *       California at Berkeley, USA
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEVR(char jobz, char range, int n, double* d, double* e, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, int* isuppz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSTEVR(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSTEVX computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric tridiagonal matrix A.  Eigenvalues and
 *  eigenvectors can be selected by specifying either a range of values
 *  or a range of indices for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  N       (input) INTEGER
 *          The order of the matrix.  N >= 0.
 *
 *  D       (input/output) DOUBLE PRECISION array, dimension (N)
 *          On entry, the n diagonal elements of the tridiagonal matrix
 *          A.
 *          On exit, D may be multiplied by a constant factor chosen
 *          to avoid over/underflow in computing the eigenvalues.
 *
 *  E       (input/output) DOUBLE PRECISION array, dimension (max(1,N-1))
 *          On entry, the (n-1) subdiagonal elements of the tridiagonal
 *          matrix A in elements 1 to N-1 of E.
 *          On exit, E may be multiplied by a constant factor chosen
 *          to avoid over/underflow in computing the eigenvalues.
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less
 *          than or equal to zero, then  EPS*|T|  will be used in
 *          its place, where |T| is the 1-norm of the tridiagonal
 *          matrix.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *          See "Computing Small Singular Values of Bidiagonal Matrices
 *          with Guaranteed High Relative Accuracy," by Demmel and
 *          Kahan, LAPACK Working Note #3.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          The first M elements contain the selected eigenvalues in
 *          ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M) )
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          If an eigenvector fails to converge (INFO > 0), then that
 *          column of Z contains the latest approximation to the
 *          eigenvector, and the index of the eigenvector is returned
 *          in IFAIL.  If JOBZ = 'N', then Z is not referenced.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (5*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (N)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvectors that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, then i eigenvectors failed to converge.
 *                Their indices are stored in array IFAIL.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSTEVX(char jobz, char range, int n, double* d, double* e, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int* iwork, int* ifail)
{
    int info;
    ::F_DSTEVX(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYCON estimates the reciprocal of the condition number (in the
 *  1-norm) of a real symmetric matrix A using the factorization
 *  A = U*D*U**T or A = L*D*L**T computed by DSYTRF.
 *
 *  An estimate is obtained for norm(inv(A)), and the reciprocal of the
 *  condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by DSYTRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSYTRF.
 *
 *  ANORM   (input) DOUBLE PRECISION
 *          The 1-norm of the original matrix A.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYCON(char uplo, int n, double* a, int lda, int* ipiv, double anorm, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DSYCON(&uplo, &n, a, &lda, ipiv, &anorm, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYEVR computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric matrix A.  Eigenvalues and eigenvectors can be
 *  selected by specifying either a range of values or a range of
 *  indices for the desired eigenvalues.
 *
 *  DSYEVR first reduces the matrix A to tridiagonal form T with a call
 *  to DSYTRD.  Then, whenever possible, DSYEVR calls DSTEMR to compute
 *  the eigenspectrum using Relatively Robust Representations.  DSTEMR
 *  computes eigenvalues by the dqds algorithm, while orthogonal
 *  eigenvectors are computed from various "good" L D L^T representations
 *  (also known as Relatively Robust Representations). Gram-Schmidt
 *  orthogonalization is avoided as far as possible. More specifically,
 *  the various steps of the algorithm are as follows.
 *
 *  For each unreduced block (submatrix) of T,
 *     (a) Compute T - sigma I  = L D L^T, so that L and D
 *         define all the wanted eigenvalues to high relative accuracy.
 *         This means that small relative changes in the entries of D and L
 *         cause only small relative changes in the eigenvalues and
 *         eigenvectors. The standard (unfactored) representation of the
 *         tridiagonal matrix T does not have this property in general.
 *     (b) Compute the eigenvalues to suitable accuracy.
 *         If the eigenvectors are desired, the algorithm attains full
 *         accuracy of the computed eigenvalues only right before
 *         the corresponding vectors have to be computed, see steps c) and d).
 *     (c) For each cluster of close eigenvalues, select a new
 *         shift close to the cluster, find a new factorization, and refine
 *         the shifted eigenvalues to suitable accuracy.
 *     (d) For each eigenvalue with a large enough relative separation compute
 *         the corresponding eigenvector by forming a rank revealing twisted
 *         factorization. Go back to (c) for any clusters that remain.
 *
 *  The desired accuracy of the output can be specified by the input
 *  parameter ABSTOL.
 *
 *  For more details, see DSTEMR's documentation and:
 *  - Inderjit S. Dhillon and Beresford N. Parlett: "Multiple representations
 *    to compute orthogonal eigenvectors of symmetric tridiagonal matrices,"
 *    Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004.
 *  - Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and
 *    Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25,
 *    2004.  Also LAPACK Working Note 154.
 *  - Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric
 *    tridiagonal eigenvalue/eigenvector problem",
 *    Computer Science Division Technical Report No. UCB/CSD-97-971,
 *    UC Berkeley, May 1997.
 *
 *
 *  Note 1 : DSYEVR calls DSTEMR when the full spectrum is requested
 *  on machines which conform to the ieee-754 floating point standard.
 *  DSYEVR calls DSTEBZ and SSTEIN on non-ieee machines and
 *  when partial spectrum requests are made.
 *
 *  Normal execution of DSTEMR may create NaNs and infinities and
 *  hence may abort due to a floating point exception in environments
 *  which do not handle NaNs and infinities in the ieee standard default
 *  manner.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 ********** For RANGE = 'V' or 'I' and IU - IL < N - 1, DSTEBZ and
 ********** DSTEIN are called
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of A contains the
 *          upper triangular part of the matrix A.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of A contains
 *          the lower triangular part of the matrix A.
 *          On exit, the lower triangle (if UPLO='L') or the upper
 *          triangle (if UPLO='U') of A, including the diagonal, is
 *          destroyed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing A to tridiagonal form.
 *
 *          See "Computing Small Singular Values of Bidiagonal Matrices
 *          with Guaranteed High Relative Accuracy," by Demmel and
 *          Kahan, LAPACK Working Note #3.
 *
 *          If high relative accuracy is important, set ABSTOL to
 *          DLAMCH( 'Safe minimum' ).  Doing so will guarantee that
 *          eigenvalues are computed to high relative accuracy when
 *          possible in future releases.  The current code does not
 *          make any guarantees about high relative accuracy, but
 *          future releases will. See J. Barlow and J. Demmel,
 *          "Computing Accurate Eigensystems of Scaled Diagonally
 *          Dominant Matrices", LAPACK Working Note #7, for a discussion
 *          of which matrices define their eigenvalues to high relative
 *          accuracy.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          The first M elements contain the selected eigenvalues in
 *          ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          If JOBZ = 'N', then Z is not referenced.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *          Supplying N columns is always safe.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  ISUPPZ  (output) INTEGER array, dimension ( 2*max(1,M) )
 *          The support of the eigenvectors in Z, i.e., the indices
 *          indicating the nonzero elements in Z. The i-th eigenvector
 *          is nonzero only in elements ISUPPZ( 2*i-1 ) through
 *          ISUPPZ( 2*i ).
 ********** Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,26*N).
 *          For optimal efficiency, LWORK >= (NB+6)*N,
 *          where NB is the max of the blocksize for DSYTRD and DORMTR
 *          returned by ILAENV.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.  LIWORK >= max(1,10*N).
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal size of the IWORK array,
 *          returns this value as the first entry of the IWORK array, and
 *          no error message related to LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  Internal error
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Inderjit Dhillon, IBM Almaden, USA
 *     Osni Marques, LBNL/NERSC, USA
 *     Ken Stanley, Computer Science Division, University of
 *       California at Berkeley, USA
 *     Jason Riedy, Computer Science Division, University of
 *       California at Berkeley, USA
 *
 * =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYEVR(char jobz, char range, char uplo, int n, double* a, int lda, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, int* isuppz, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSYEVR(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYEVX computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric matrix A.  Eigenvalues and eigenvectors can be
 *  selected by specifying either a range of values or a range of indices
 *  for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of A contains the
 *          upper triangular part of the matrix A.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of A contains
 *          the lower triangular part of the matrix A.
 *          On exit, the lower triangle (if UPLO='L') or the upper
 *          triangle (if UPLO='U') of A, including the diagonal, is
 *          destroyed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing A to tridiagonal form.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *          See "Computing Small Singular Values of Bidiagonal Matrices
 *          with Guaranteed High Relative Accuracy," by Demmel and
 *          Kahan, LAPACK Working Note #3.
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          On normal exit, the first M elements contain the selected
 *          eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          If an eigenvector fails to converge, then that column of Z
 *          contains the latest approximation to the eigenvector, and the
 *          index of the eigenvector is returned in IFAIL.
 *          If JOBZ = 'N', then Z is not referenced.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of the array WORK.  LWORK >= 1, when N <= 1;
 *          otherwise 8*N.
 *          For optimal efficiency, LWORK >= (NB+3)*N,
 *          where NB is the max of the blocksize for DSYTRD and DORMTR
 *          returned by ILAENV.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (N)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvectors that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, then i eigenvectors failed to converge.
 *                Their indices are stored in array IFAIL.
 *
 * =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYEVX(char jobz, char range, char uplo, int n, double* a, int lda, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int lwork, int* iwork, int* ifail)
{
    int info;
    ::F_DSYEVX(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYGST reduces a real symmetric-definite generalized eigenproblem
 *  to standard form.
 *
 *  If ITYPE = 1, the problem is A*x = lambda*B*x,
 *  and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
 *
 *  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 *  B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
 *
 *  B must have been previously factorized as U**T*U or L*L**T by DPOTRF.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
 *          = 2 or 3: compute U*A*U**T or L**T*A*L.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored and B is factored as
 *                  U**T*U;
 *          = 'L':  Lower triangle of A is stored and B is factored as
 *                  L*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, if INFO = 0, the transformed matrix, stored in the
 *          same format as A.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,N)
 *          The triangular factor from the Cholesky factorization of B,
 *          as returned by DPOTRF.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYGST(int itype, char uplo, int n, double* a, int lda, double* b, int ldb)
{
    int info;
    ::F_DSYGST(&itype, &uplo, &n, a, &lda, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYGV computes all the eigenvalues, and optionally, the eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
 *  Here A and B are assumed to be symmetric and B is also
 *  positive definite.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          Specifies the problem type to be solved:
 *          = 1:  A*x = (lambda)*B*x
 *          = 2:  A*B*x = (lambda)*x
 *          = 3:  B*A*x = (lambda)*x
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of A contains the
 *          upper triangular part of the matrix A.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of A contains
 *          the lower triangular part of the matrix A.
 *
 *          On exit, if JOBZ = 'V', then if INFO = 0, A contains the
 *          matrix Z of eigenvectors.  The eigenvectors are normalized
 *          as follows:
 *          if ITYPE = 1 or 2, Z**T*B*Z = I;
 *          if ITYPE = 3, Z**T*inv(B)*Z = I.
 *          If JOBZ = 'N', then on exit the upper triangle (if UPLO='U')
 *          or the lower triangle (if UPLO='L') of A, including the
 *          diagonal, is destroyed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the symmetric positive definite matrix B.
 *          If UPLO = 'U', the leading N-by-N upper triangular part of B
 *          contains the upper triangular part of the matrix B.
 *          If UPLO = 'L', the leading N-by-N lower triangular part of B
 *          contains the lower triangular part of the matrix B.
 *
 *          On exit, if INFO <= N, the part of B containing the matrix is
 *          overwritten by the triangular factor U or L from the Cholesky
 *          factorization B = U**T*U or B = L*L**T.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of the array WORK.  LWORK >= max(1,3*N-1).
 *          For optimal efficiency, LWORK >= (NB+2)*N,
 *          where NB is the blocksize for DSYTRD returned by ILAENV.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  DPOTRF or DSYEV returned an error code:
 *             <= N:  if INFO = i, DSYEV failed to converge;
 *                    i off-diagonal elements of an intermediate
 *                    tridiagonal form did not converge to zero;
 *             > N:   if INFO = N + i, for 1 <= i <= N, then the leading
 *                    minor of order i of B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYGV(int itype, char jobz, char uplo, int n, double* a, int lda, double* b, int ldb, double* w, double* work, int lwork)
{
    int info;
    ::F_DSYGV(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYGVD computes all the eigenvalues, and optionally, the eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
 *  B are assumed to be symmetric and B is also positive definite.
 *  If eigenvectors are desired, it uses a divide and conquer algorithm.
 *
 *  The divide and conquer algorithm makes very mild assumptions about
 *  floating point arithmetic. It will work on machines with a guard
 *  digit in add/subtract, or on those binary machines without guard
 *  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 *  Cray-2. It could conceivably fail on hexadecimal or decimal machines
 *  without guard digits, but we know of none.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          Specifies the problem type to be solved:
 *          = 1:  A*x = (lambda)*B*x
 *          = 2:  A*B*x = (lambda)*x
 *          = 3:  B*A*x = (lambda)*x
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangles of A and B are stored;
 *          = 'L':  Lower triangles of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of A contains the
 *          upper triangular part of the matrix A.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of A contains
 *          the lower triangular part of the matrix A.
 *
 *          On exit, if JOBZ = 'V', then if INFO = 0, A contains the
 *          matrix Z of eigenvectors.  The eigenvectors are normalized
 *          as follows:
 *          if ITYPE = 1 or 2, Z**T*B*Z = I;
 *          if ITYPE = 3, Z**T*inv(B)*Z = I.
 *          If JOBZ = 'N', then on exit the upper triangle (if UPLO='U')
 *          or the lower triangle (if UPLO='L') of A, including the
 *          diagonal, is destroyed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the symmetric matrix B.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of B contains the
 *          upper triangular part of the matrix B.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of B contains
 *          the lower triangular part of the matrix B.
 *
 *          On exit, if INFO <= N, the part of B containing the matrix is
 *          overwritten by the triangular factor U or L from the Cholesky
 *          factorization B = U**T*U or B = L*L**T.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          If INFO = 0, the eigenvalues in ascending order.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If N <= 1,               LWORK >= 1.
 *          If JOBZ = 'N' and N > 1, LWORK >= 2*N+1.
 *          If JOBZ = 'V' and N > 1, LWORK >= 1 + 6*N + 2*N**2.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal sizes of the WORK and IWORK
 *          arrays, returns these values as the first entries of the WORK
 *          and IWORK arrays, and no error message related to LWORK or
 *          LIWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If N <= 1,                LIWORK >= 1.
 *          If JOBZ  = 'N' and N > 1, LIWORK >= 1.
 *          If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal sizes of the WORK and
 *          IWORK arrays, returns these values as the first entries of
 *          the WORK and IWORK arrays, and no error message related to
 *          LWORK or LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  DPOTRF or DSYEVD returned an error code:
 *             <= N:  if INFO = i and JOBZ = 'N', then the algorithm
 *                    failed to converge; i off-diagonal elements of an
 *                    intermediate tridiagonal form did not converge to
 *                    zero;
 *                    if INFO = i and JOBZ = 'V', then the algorithm
 *                    failed to compute an eigenvalue while working on
 *                    the submatrix lying in rows and columns INFO/(N+1)
 *                    through mod(INFO,N+1);
 *             > N:   if INFO = N + i, for 1 <= i <= N, then the leading
 *                    minor of order i of B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
 *
 *  Modified so that no backsubstitution is performed if DSYEVD fails to
 *  converge (NEIG in old code could be greater than N causing out of
 *  bounds reference to A - reported by Ralf Meyer).  Also corrected the
 *  description of INFO and the test on ITYPE. Sven, 16 Feb 05.
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYGVD(int itype, char jobz, char uplo, int n, double* a, int lda, double* b, int ldb, double* w, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSYGVD(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYGVX computes selected eigenvalues, and optionally, eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
 *  and B are assumed to be symmetric and B is also positive definite.
 *  Eigenvalues and eigenvectors can be selected by specifying either a
 *  range of values or a range of indices for the desired eigenvalues.
 *
 *  Arguments
 *  =========
 *
 *  ITYPE   (input) INTEGER
 *          Specifies the problem type to be solved:
 *          = 1:  A*x = (lambda)*B*x
 *          = 2:  A*B*x = (lambda)*x
 *          = 3:  B*A*x = (lambda)*x
 *
 *  JOBZ    (input) CHARACTER*1
 *          = 'N':  Compute eigenvalues only;
 *          = 'V':  Compute eigenvalues and eigenvectors.
 *
 *  RANGE   (input) CHARACTER*1
 *          = 'A': all eigenvalues will be found.
 *          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                 will be found.
 *          = 'I': the IL-th through IU-th eigenvalues will be found.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A and B are stored;
 *          = 'L':  Lower triangle of A and B are stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix pencil (A,B).  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of A contains the
 *          upper triangular part of the matrix A.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of A contains
 *          the lower triangular part of the matrix A.
 *
 *          On exit, the lower triangle (if UPLO='L') or the upper
 *          triangle (if UPLO='U') of A, including the diagonal, is
 *          destroyed.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB, N)
 *          On entry, the symmetric matrix B.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of B contains the
 *          upper triangular part of the matrix B.  If UPLO = 'L',
 *          the leading N-by-N lower triangular part of B contains
 *          the lower triangular part of the matrix B.
 *
 *          On exit, if INFO <= N, the part of B containing the matrix is
 *          overwritten by the triangular factor U or L from the Cholesky
 *          factorization B = U**T*U or B = L*L**T.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION
 *  VU      (input) DOUBLE PRECISION
 *          If RANGE='V', the lower and upper bounds of the interval to
 *          be searched for eigenvalues. VL < VU.
 *          Not referenced if RANGE = 'A' or 'I'.
 *
 *  IL      (input) INTEGER
 *  IU      (input) INTEGER
 *          If RANGE='I', the indices (in ascending order) of the
 *          smallest and largest eigenvalues to be returned.
 *          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
 *          Not referenced if RANGE = 'A' or 'V'.
 *
 *  ABSTOL  (input) DOUBLE PRECISION
 *          The absolute error tolerance for the eigenvalues.
 *          An approximate eigenvalue is accepted as converged
 *          when it is determined to lie in an interval [a,b]
 *          of width less than or equal to
 *
 *                  ABSTOL + EPS *   max( |a|,|b| ) ,
 *
 *          where EPS is the machine precision.  If ABSTOL is less than
 *          or equal to zero, then  EPS*|T|  will be used in its place,
 *          where |T| is the 1-norm of the tridiagonal matrix obtained
 *          by reducing A to tridiagonal form.
 *
 *          Eigenvalues will be computed most accurately when ABSTOL is
 *          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
 *          If this routine returns with INFO>0, indicating that some
 *          eigenvectors did not converge, try setting ABSTOL to
 *          2*DLAMCH('S').
 *
 *  M       (output) INTEGER
 *          The total number of eigenvalues found.  0 <= M <= N.
 *          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
 *
 *  W       (output) DOUBLE PRECISION array, dimension (N)
 *          On normal exit, the first M elements contain the selected
 *          eigenvalues in ascending order.
 *
 *  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))
 *          If JOBZ = 'N', then Z is not referenced.
 *          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
 *          contain the orthonormal eigenvectors of the matrix A
 *          corresponding to the selected eigenvalues, with the i-th
 *          column of Z holding the eigenvector associated with W(i).
 *          The eigenvectors are normalized as follows:
 *          if ITYPE = 1 or 2, Z**T*B*Z = I;
 *          if ITYPE = 3, Z**T*inv(B)*Z = I.
 *
 *          If an eigenvector fails to converge, then that column of Z
 *          contains the latest approximation to the eigenvector, and the
 *          index of the eigenvector is returned in IFAIL.
 *          Note: the user must ensure that at least max(1,M) columns are
 *          supplied in the array Z; if RANGE = 'V', the exact value of M
 *          is not known in advance and an upper bound must be used.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z.  LDZ >= 1, and if
 *          JOBZ = 'V', LDZ >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of the array WORK.  LWORK >= max(1,8*N).
 *          For optimal efficiency, LWORK >= (NB+3)*N,
 *          where NB is the blocksize for DSYTRD returned by ILAENV.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (5*N)
 *
 *  IFAIL   (output) INTEGER array, dimension (N)
 *          If JOBZ = 'V', then if INFO = 0, the first M elements of
 *          IFAIL are zero.  If INFO > 0, then IFAIL contains the
 *          indices of the eigenvectors that failed to converge.
 *          If JOBZ = 'N', then IFAIL is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  DPOTRF or DSYEVX returned an error code:
 *             <= N:  if INFO = i, DSYEVX failed to converge;
 *                    i eigenvectors failed to converge.  Their indices
 *                    are stored in array IFAIL.
 *             > N:   if INFO = N + i, for 1 <= i <= N, then the leading
 *                    minor of order i of B is not positive definite.
 *                    The factorization of B could not be completed and
 *                    no eigenvalues or eigenvectors were computed.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA
 *
 * =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYGVX(int itype, char jobz, char range, char uplo, int n, double* a, int lda, double* b, int ldb, double vl, double vu, int il, int iu, double abstol, int* m, double* w, double* z, int ldz, double* work, int lwork, int* iwork, int* ifail)
{
    int info;
    ::F_DSYGVX(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYRFS improves the computed solution to a system of linear
 *  equations when the coefficient matrix is symmetric indefinite, and
 *  provides error bounds and backward error estimates for the solution.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The symmetric matrix A.  If UPLO = 'U', the leading N-by-N
 *          upper triangular part of A contains the upper triangular part
 *          of the matrix A, and the strictly lower triangular part of A
 *          is not referenced.  If UPLO = 'L', the leading N-by-N lower
 *          triangular part of A contains the lower triangular part of
 *          the matrix A, and the strictly upper triangular part of A is
 *          not referenced.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  AF      (input) DOUBLE PRECISION array, dimension (LDAF,N)
 *          The factored form of the matrix A.  AF contains the block
 *          diagonal matrix D and the multipliers used to obtain the
 *          factor U or L from the factorization A = U*D*U**T or
 *          A = L*D*L**T as computed by DSYTRF.
 *
 *  LDAF    (input) INTEGER
 *          The leading dimension of the array AF.  LDAF >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSYTRF.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input/output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          On entry, the solution matrix X, as computed by DSYTRS.
 *          On exit, the improved solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Internal Parameters
 *  ===================
 *
 *  ITMAX is the maximum number of steps of iterative refinement.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYRFS(char uplo, int n, int nrhs, double* a, int lda, double* af, int ldaf, int* ipiv, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DSYRFS(&uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYSV computes the solution to a real system of linear equations
 *     A * X = B,
 *  where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
 *  matrices.
 *
 *  The diagonal pivoting method is used to factor A as
 *     A = U * D * U**T,  if UPLO = 'U', or
 *     A = L * D * L**T,  if UPLO = 'L',
 *  where U (or L) is a product of permutation and unit upper (lower)
 *  triangular matrices, and D is symmetric and block diagonal with
 *  1-by-1 and 2-by-2 diagonal blocks.  The factored form of A is then
 *  used to solve the system of equations A * X = B.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, if INFO = 0, the block diagonal matrix D and the
 *          multipliers used to obtain the factor U or L from the
 *          factorization A = U*D*U**T or A = L*D*L**T as computed by
 *          DSYTRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D, as
 *          determined by DSYTRF.  If IPIV(k) > 0, then rows and columns
 *          k and IPIV(k) were interchanged, and D(k,k) is a 1-by-1
 *          diagonal block.  If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0,
 *          then rows and columns k-1 and -IPIV(k) were interchanged and
 *          D(k-1:k,k-1:k) is a 2-by-2 diagonal block.  If UPLO = 'L' and
 *          IPIV(k) = IPIV(k+1) < 0, then rows and columns k+1 and
 *          -IPIV(k) were interchanged and D(k:k+1,k:k+1) is a 2-by-2
 *          diagonal block.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of WORK.  LWORK >= 1, and for best performance
 *          LWORK >= max(1,N*NB), where NB is the optimal blocksize for
 *          DSYTRF.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, D(i,i) is exactly zero.  The factorization
 *               has been completed, but the block diagonal matrix D is
 *               exactly singular, so the solution could not be computed.
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DSYSV(char uplo, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, double* work, int lwork)
{
    int info;
    ::F_DSYSV(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYSVX uses the diagonal pivoting factorization to compute the
 *  solution to a real system of linear equations A * X = B,
 *  where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
 *  matrices.
 *
 *  Error bounds on the solution and a condition estimate are also
 *  provided.
 *
 *  Description
 *  ===========
 *
 *  The following steps are performed:
 *
 *  1. If FACT = 'N', the diagonal pivoting method is used to factor A.
 *     The form of the factorization is
 *        A = U * D * U**T,  if UPLO = 'U', or
 *        A = L * D * L**T,  if UPLO = 'L',
 *     where U (or L) is a product of permutation and unit upper (lower)
 *     triangular matrices, and D is symmetric and block diagonal with
 *     1-by-1 and 2-by-2 diagonal blocks.
 *
 *  2. If some D(i,i)=0, so that D is exactly singular, then the routine
 *     returns with INFO = i. Otherwise, the factored form of A is used
 *     to estimate the condition number of the matrix A.  If the
 *     reciprocal of the condition number is less than machine precision,
 *  C++ Return value: INFO    (output) INTEGER
 *     to solve for X and compute error bounds as described below.
 *
 *  3. The system of equations is solved for X using the factored form
 *     of A.
 *
 *  4. Iterative refinement is applied to improve the computed solution
 *     matrix and calculate error bounds and backward error estimates
 *     for it.
 *
 *  Arguments
 *  =========
 *
 *  FACT    (input) CHARACTER*1
 *          Specifies whether or not the factored form of A has been
 *          supplied on entry.
 *          = 'F':  On entry, AF and IPIV contain the factored form of
 *                  A.  AF and IPIV will not be modified.
 *          = 'N':  The matrix A will be copied to AF and factored.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The number of linear equations, i.e., the order of the
 *          matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The symmetric matrix A.  If UPLO = 'U', the leading N-by-N
 *          upper triangular part of A contains the upper triangular part
 *          of the matrix A, and the strictly lower triangular part of A
 *          is not referenced.  If UPLO = 'L', the leading N-by-N lower
 *          triangular part of A contains the lower triangular part of
 *          the matrix A, and the strictly upper triangular part of A is
 *          not referenced.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  AF      (input or output) DOUBLE PRECISION array, dimension (LDAF,N)
 *          If FACT = 'F', then AF is an input argument and on entry
 *          contains the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L from the factorization
 *          A = U*D*U**T or A = L*D*L**T as computed by DSYTRF.
 *
 *          If FACT = 'N', then AF is an output argument and on exit
 *          returns the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L from the factorization
 *          A = U*D*U**T or A = L*D*L**T.
 *
 *  LDAF    (input) INTEGER
 *          The leading dimension of the array AF.  LDAF >= max(1,N).
 *
 *  IPIV    (input or output) INTEGER array, dimension (N)
 *          If FACT = 'F', then IPIV is an input argument and on entry
 *          contains details of the interchanges and the block structure
 *          of D, as determined by DSYTRF.
 *          If IPIV(k) > 0, then rows and columns k and IPIV(k) were
 *          interchanged and D(k,k) is a 1-by-1 diagonal block.
 *          If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and
 *          columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
 *          is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =
 *          IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
 *          interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
 *
 *          If FACT = 'N', then IPIV is an output argument and on exit
 *          contains details of the interchanges and the block structure
 *          of D, as determined by DSYTRF.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The N-by-NRHS right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The estimate of the reciprocal condition number of the matrix
 *          A.  If RCOND is less than the machine precision (in
 *          particular, if RCOND = 0), the matrix is singular to working
 *          precision.  This condition is indicated by a return code of
 *  C++ Return value: INFO    (output) INTEGER
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of WORK.  LWORK >= max(1,3*N), and for best
 *          performance, when FACT = 'N', LWORK >= max(1,3*N,N*NB), where
 *          NB is the optimal blocksize for DSYTRF.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, and i is
 *                <= N:  D(i,i) is exactly zero.  The factorization
 *                       has been completed but the factor D is exactly
 *                       singular, so the solution and error bounds could
 *                       not be computed. RCOND = 0 is returned.
 *                = N+1: D is nonsingular, but RCOND is less than machine
 *                       precision, meaning that the matrix is singular
 *                       to working precision.  Nevertheless, the
 *                       solution and error bounds are computed because
 *                       there are a number of situations where the
 *                       computed solution can be more accurate than the
 *                       value of RCOND would suggest.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYSVX(char fact, char uplo, int n, int nrhs, double* a, int lda, double* af, int ldaf, int* ipiv, double* b, int ldb, double* x, int ldx, double* rcond)
{
    int info;
    ::F_DSYSVX(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, rcond, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYTRD reduces a real symmetric matrix A to real symmetric
 *  tridiagonal form T by an orthogonal similarity transformation:
 *  Q**T * A * Q = T.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *          On exit, if UPLO = 'U', the diagonal and first superdiagonal
 *          of A are overwritten by the corresponding elements of the
 *          tridiagonal matrix T, and the elements above the first
 *          superdiagonal, with the array TAU, represent the orthogonal
 *          matrix Q as a product of elementary reflectors; if UPLO
 *          = 'L', the diagonal and first subdiagonal of A are over-
 *          written by the corresponding elements of the tridiagonal
 *          matrix T, and the elements below the first subdiagonal, with
 *          the array TAU, represent the orthogonal matrix Q as a product
 *          of elementary reflectors. See Further Details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  D       (output) DOUBLE PRECISION array, dimension (N)
 *          The diagonal elements of the tridiagonal matrix T:
 *          D(i) = A(i,i).
 *
 *  E       (output) DOUBLE PRECISION array, dimension (N-1)
 *          The off-diagonal elements of the tridiagonal matrix T:
 *          E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (N-1)
 *          The scalar factors of the elementary reflectors (see Further
 *          Details).
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= 1.
 *          For optimum performance LWORK >= N*NB, where NB is the
 *          optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  If UPLO = 'U', the matrix Q is represented as a product of elementary
 *  reflectors
 *
 *     Q = H(n-1) . . . H(2) H(1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
 *  A(1:i-1,i+1), and tau in TAU(i).
 *
 *  If UPLO = 'L', the matrix Q is represented as a product of elementary
 *  reflectors
 *
 *     Q = H(1) H(2) . . . H(n-1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v'
 *
 *  where tau is a real scalar, and v is a real vector with
 *  v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
 *  and tau in TAU(i).
 *
 *  The contents of A on exit are illustrated by the following examples
 *  with n = 5:
 *
 *  if UPLO = 'U':                       if UPLO = 'L':
 *
 *    (  d   e   v2  v3  v4 )              (  d                  )
 *    (      d   e   v3  v4 )              (  e   d              )
 *    (          d   e   v4 )              (  v1  e   d          )
 *    (              d   e  )              (  v1  v2  e   d      )
 *    (                  d  )              (  v1  v2  v3  e   d  )
 *
 *  where d and e denote diagonal and off-diagonal elements of T, and vi
 *  denotes an element of the vector defining H(i).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYTRD(char uplo, int n, double* a, int lda, double* d, double* e, double* tau, double* work, int lwork)
{
    int info;
    ::F_DSYTRD(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYTRF computes the factorization of a real symmetric matrix A using
 *  the Bunch-Kaufman diagonal pivoting method.  The form of the
 *  factorization is
 *
 *     A = U*D*U**T  or  A = L*D*L**T
 *
 *  where U (or L) is a product of permutation and unit upper (lower)
 *  triangular matrices, and D is symmetric and block diagonal with
 *  1-by-1 and 2-by-2 diagonal blocks.
 *
 *  This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, the block diagonal matrix D and the multipliers used
 *          to obtain the factor U or L (see below for further details).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (output) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D.
 *          If IPIV(k) > 0, then rows and columns k and IPIV(k) were
 *          interchanged and D(k,k) is a 1-by-1 diagonal block.
 *          If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and
 *          columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
 *          is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =
 *          IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
 *          interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The length of WORK.  LWORK >=1.  For best performance
 *          LWORK >= N*NB, where NB is the block size returned by ILAENV.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, D(i,i) is exactly zero.  The factorization
 *                has been completed, but the block diagonal matrix D is
 *                exactly singular, and division by zero will occur if it
 *                is used to solve a system of equations.
 *
 *  Further Details
 *  ===============
 *
 *  If UPLO = 'U', then A = U*D*U', where
 *     U = P(n)*U(n)* ... *P(k)U(k)* ...,
 *  i.e., U is a product of terms P(k)*U(k), where k decreases from n to
 *  1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
 *  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
 *  defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
 *  that if the diagonal block D(k) is of order s (s = 1 or 2), then
 *
 *             (   I    v    0   )   k-s
 *     U(k) =  (   0    I    0   )   s
 *             (   0    0    I   )   n-k
 *                k-s   s   n-k
 *
 *  If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
 *  If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
 *  and A(k,k), and v overwrites A(1:k-2,k-1:k).
 *
 *  If UPLO = 'L', then A = L*D*L', where
 *     L = P(1)*L(1)* ... *P(k)*L(k)* ...,
 *  i.e., L is a product of terms P(k)*L(k), where k increases from 1 to
 *  n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
 *  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
 *  defined by IPIV(k), and L(k) is a unit lower triangular matrix, such
 *  that if the diagonal block D(k) is of order s (s = 1 or 2), then
 *
 *             (   I    0     0   )  k-1
 *     L(k) =  (   0    I     0   )  s
 *             (   0    v     I   )  n-k-s+1
 *                k-1   s  n-k-s+1
 *
 *  If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
 *  If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
 *  and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
 *
 *  =====================================================================
 *
 *     .. Local Scalars ..
 **/
int C_DSYTRF(char uplo, int n, double* a, int lda, int* ipiv, double* work, int lwork)
{
    int info;
    ::F_DSYTRF(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYTRI computes the inverse of a real symmetric indefinite matrix
 *  A using the factorization A = U*D*U**T or A = L*D*L**T computed by
 *  DSYTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the block diagonal matrix D and the multipliers
 *          used to obtain the factor U or L as computed by DSYTRF.
 *
 *          On exit, if INFO = 0, the (symmetric) inverse of the original
 *          matrix.  If UPLO = 'U', the upper triangular part of the
 *          inverse is formed and the part of A below the diagonal is not
 *          referenced; if UPLO = 'L' the lower triangular part of the
 *          inverse is formed and the part of A above the diagonal is
 *          not referenced.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSYTRF.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, D(i,i) = 0; the matrix is singular and its
 *               inverse could not be computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYTRI(char uplo, int n, double* a, int lda, int* ipiv, double* work)
{
    int info;
    ::F_DSYTRI(&uplo, &n, a, &lda, ipiv, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DSYTRS solves a system of linear equations A*X = B with a real
 *  symmetric matrix A using the factorization A = U*D*U**T or
 *  A = L*D*L**T computed by DSYTRF.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by DSYTRF.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  IPIV    (input) INTEGER array, dimension (N)
 *          Details of the interchanges and the block structure of D
 *          as determined by DSYTRF.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DSYTRS(char uplo, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DSYTRS(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTBCON estimates the reciprocal of the condition number of a
 *  triangular band matrix A, in either the 1-norm or the infinity-norm.
 *
 *  The norm of A is computed and an estimate is obtained for
 *  norm(inv(A)), then the reciprocal of the condition number is
 *  computed as
 *     RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 *  Arguments
 *  =========
 *
 *  NORM    (input) CHARACTER*1
 *          Specifies whether the 1-norm condition number or the
 *          infinity-norm condition number is required:
 *          = '1' or 'O':  1-norm;
 *          = 'I':         Infinity-norm.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals or subdiagonals of the
 *          triangular band matrix A.  KD >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The upper or lower triangular band matrix A, stored in the
 *          first kd+1 rows of the array. The j-th column of A is stored
 *          in the j-th column of the array AB as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *          If DIAG = 'U', the diagonal elements of A are not referenced
 *          and are assumed to be 1.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(norm(A) * norm(inv(A))).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTBCON(char norm, char uplo, char diag, int n, int kd, double* ab, int ldab, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DTBCON(&norm, &uplo, &diag, &n, &kd, ab, &ldab, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTBRFS provides error bounds and backward error estimates for the
 *  solution to a system of linear equations with a triangular band
 *  coefficient matrix.
 *
 *  The solution matrix X must be computed by DTBTRS or some other
 *  means before entering this routine.  DTBRFS does not do iterative
 *  refinement because doing so cannot improve the backward error.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals or subdiagonals of the
 *          triangular band matrix A.  KD >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The upper or lower triangular band matrix A, stored in the
 *          first kd+1 rows of the array. The j-th column of A is stored
 *          in the j-th column of the array AB as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *          If DIAG = 'U', the diagonal elements of A are not referenced
 *          and are assumed to be 1.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          The solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTBRFS(char uplo, char trans, char diag, int n, int kd, int nrhs, double* ab, int ldab, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DTBRFS(&uplo, &trans, &diag, &n, &kd, &nrhs, ab, &ldab, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTBTRS solves a triangular system of the form
 *
 *     A * X = B  or  A**T * X = B,
 *
 *  where A is a triangular band matrix of order N, and B is an
 *  N-by NRHS matrix.  A check is made to verify that A is nonsingular.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  KD      (input) INTEGER
 *          The number of superdiagonals or subdiagonals of the
 *          triangular band matrix A.  KD >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AB      (input) DOUBLE PRECISION array, dimension (LDAB,N)
 *          The upper or lower triangular band matrix A, stored in the
 *          first kd+1 rows of AB.  The j-th column of A is stored
 *          in the j-th column of the array AB as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *          If DIAG = 'U', the diagonal elements of A are not referenced
 *          and are assumed to be 1.
 *
 *  LDAB    (input) INTEGER
 *          The leading dimension of the array AB.  LDAB >= KD+1.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, if INFO = 0, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the i-th diagonal element of A is zero,
 *                indicating that the matrix is singular and the
 *                solutions X have not been computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTBTRS(char uplo, char trans, char diag, int n, int kd, int nrhs, double* ab, int ldab, double* b, int ldb)
{
    int info;
    ::F_DTBTRS(&uplo, &trans, &diag, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTGEVC computes some or all of the right and/or left eigenvectors of
 *  a pair of real matrices (S,P), where S is a quasi-triangular matrix
 *  and P is upper triangular.  Matrix pairs of this type are produced by
 *  the generalized Schur factorization of a matrix pair (A,B):
 *
 *     A = Q*S*Z**T,  B = Q*P*Z**T
 *
 *  as computed by DGGHRD + DHGEQZ.
 *
 *  The right eigenvector x and the left eigenvector y of (S,P)
 *  corresponding to an eigenvalue w are defined by:
 *
 *     S*x = w*P*x,  (y**H)*S = w*(y**H)*P,
 *
 *  where y**H denotes the conjugate tranpose of y.
 *  The eigenvalues are not input to this routine, but are computed
 *  directly from the diagonal blocks of S and P.
 *
 *  This routine returns the matrices X and/or Y of right and left
 *  eigenvectors of (S,P), or the products Z*X and/or Q*Y,
 *  where Z and Q are input matrices.
 *  If Q and Z are the orthogonal factors from the generalized Schur
 *  factorization of a matrix pair (A,B), then Z*X and Q*Y
 *  are the matrices of right and left eigenvectors of (A,B).
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'R': compute right eigenvectors only;
 *          = 'L': compute left eigenvectors only;
 *          = 'B': compute both right and left eigenvectors.
 *
 *  HOWMNY  (input) CHARACTER*1
 *          = 'A': compute all right and/or left eigenvectors;
 *          = 'B': compute all right and/or left eigenvectors,
 *                 backtransformed by the matrices in VR and/or VL;
 *          = 'S': compute selected right and/or left eigenvectors,
 *                 specified by the logical array SELECT.
 *
 *  SELECT  (input) LOGICAL array, dimension (N)
 *          If HOWMNY='S', SELECT specifies the eigenvectors to be
 *          computed.  If w(j) is a real eigenvalue, the corresponding
 *          real eigenvector is computed if SELECT(j) is .TRUE..
 *          If w(j) and w(j+1) are the real and imaginary parts of a
 *          complex eigenvalue, the corresponding complex eigenvector
 *          is computed if either SELECT(j) or SELECT(j+1) is .TRUE.,
 *          and on exit SELECT(j) is set to .TRUE. and SELECT(j+1) is
 *          set to .FALSE..
 *          Not referenced if HOWMNY = 'A' or 'B'.
 *
 *  N       (input) INTEGER
 *          The order of the matrices S and P.  N >= 0.
 *
 *  S       (input) DOUBLE PRECISION array, dimension (LDS,N)
 *          The upper quasi-triangular matrix S from a generalized Schur
 *          factorization, as computed by DHGEQZ.
 *
 *  LDS     (input) INTEGER
 *          The leading dimension of array S.  LDS >= max(1,N).
 *
 *  P       (input) DOUBLE PRECISION array, dimension (LDP,N)
 *          The upper triangular matrix P from a generalized Schur
 *          factorization, as computed by DHGEQZ.
 *          2-by-2 diagonal blocks of P corresponding to 2-by-2 blocks
 *          of S must be in positive diagonal form.
 *
 *  LDP     (input) INTEGER
 *          The leading dimension of array P.  LDP >= max(1,N).
 *
 *  VL      (input/output) DOUBLE PRECISION array, dimension (LDVL,MM)
 *          On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must
 *          contain an N-by-N matrix Q (usually the orthogonal matrix Q
 *          of left Schur vectors returned by DHGEQZ).
 *          On exit, if SIDE = 'L' or 'B', VL contains:
 *          if HOWMNY = 'A', the matrix Y of left eigenvectors of (S,P);
 *          if HOWMNY = 'B', the matrix Q*Y;
 *          if HOWMNY = 'S', the left eigenvectors of (S,P) specified by
 *                      SELECT, stored consecutively in the columns of
 *                      VL, in the same order as their eigenvalues.
 *
 *          A complex eigenvector corresponding to a complex eigenvalue
 *          is stored in two consecutive columns, the first holding the
 *          real part, and the second the imaginary part.
 *
 *          Not referenced if SIDE = 'R'.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of array VL.  LDVL >= 1, and if
 *          SIDE = 'L' or 'B', LDVL >= N.
 *
 *  VR      (input/output) DOUBLE PRECISION array, dimension (LDVR,MM)
 *          On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must
 *          contain an N-by-N matrix Z (usually the orthogonal matrix Z
 *          of right Schur vectors returned by DHGEQZ).
 *
 *          On exit, if SIDE = 'R' or 'B', VR contains:
 *          if HOWMNY = 'A', the matrix X of right eigenvectors of (S,P);
 *          if HOWMNY = 'B' or 'b', the matrix Z*X;
 *          if HOWMNY = 'S' or 's', the right eigenvectors of (S,P)
 *                      specified by SELECT, stored consecutively in the
 *                      columns of VR, in the same order as their
 *                      eigenvalues.
 *
 *          A complex eigenvector corresponding to a complex eigenvalue
 *          is stored in two consecutive columns, the first holding the
 *          real part and the second the imaginary part.
 *
 *          Not referenced if SIDE = 'L'.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR.  LDVR >= 1, and if
 *          SIDE = 'R' or 'B', LDVR >= N.
 *
 *  MM      (input) INTEGER
 *          The number of columns in the arrays VL and/or VR. MM >= M.
 *
 *  M       (output) INTEGER
 *          The number of columns in the arrays VL and/or VR actually
 *          used to store the eigenvectors.  If HOWMNY = 'A' or 'B', M
 *          is set to N.  Each selected real eigenvector occupies one
 *          column and each selected complex eigenvector occupies two
 *          columns.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (6*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit.
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          > 0:  the 2-by-2 block (INFO:INFO+1) does not have a complex
 *                eigenvalue.
 *
 *  Further Details
 *  ===============
 *
 *  Allocation of workspace:
 *  ---------- -- ---------
 *
 *     WORK( j ) = 1-norm of j-th column of A, above the diagonal
 *     WORK( N+j ) = 1-norm of j-th column of B, above the diagonal
 *     WORK( 2*N+1:3*N ) = real part of eigenvector
 *     WORK( 3*N+1:4*N ) = imaginary part of eigenvector
 *     WORK( 4*N+1:5*N ) = real part of back-transformed eigenvector
 *     WORK( 5*N+1:6*N ) = imaginary part of back-transformed eigenvector
 *
 *  Rowwise vs. columnwise solution methods:
 *  ------- --  ---------- -------- -------
 *
 *  Finding a generalized eigenvector consists basically of solving the
 *  singular triangular system
 *
 *   (A - w B) x = 0     (for right) or:   (A - w B)**H y = 0  (for left)
 *
 *  Consider finding the i-th right eigenvector (assume all eigenvalues
 *  are real). The equation to be solved is:
 *       n                   i
 *  0 = sum  C(j,k) v(k)  = sum  C(j,k) v(k)     for j = i,. . .,1
 *      k=j                 k=j
 *
 *  where  C = (A - w B)  (The components v(i+1:n) are 0.)
 *
 *  The "rowwise" method is:
 *
 *  (1)  v(i) := 1
 *  for j = i-1,. . .,1:
 *                          i
 *      (2) compute  s = - sum C(j,k) v(k)   and
 *                        k=j+1
 *
 *      (3) v(j) := s / C(j,j)
 *
 *  Step 2 is sometimes called the "dot product" step, since it is an
 *  inner product between the j-th row and the portion of the eigenvector
 *  that has been computed so far.
 *
 *  The "columnwise" method consists basically in doing the sums
 *  for all the rows in parallel.  As each v(j) is computed, the
 *  contribution of v(j) times the j-th column of C is added to the
 *  partial sums.  Since FORTRAN arrays are stored columnwise, this has
 *  the advantage that at each step, the elements of C that are accessed
 *  are adjacent to one another, whereas with the rowwise method, the
 *  elements accessed at a step are spaced LDS (and LDP) words apart.
 *
 *  When finding left eigenvectors, the matrix in question is the
 *  transpose of the one in storage, so the rowwise method then
 *  actually accesses columns of A and B at each step, and so is the
 *  preferred method.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTGEVC(char side, char howmny, int n, double* s, int lds, double* p, int ldp, double* vl, int ldvl, double* vr, int ldvr, int mm, int* m, double* work)
{
    int info;
    ::F_DTGEVC(&side, &howmny, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, m, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTGEXC reorders the generalized real Schur decomposition of a real
 *  matrix pair (A,B) using an orthogonal equivalence transformation
 *
 *                 (A, B) = Q * (A, B) * Z',
 *
 *  so that the diagonal block of (A, B) with row index IFST is moved
 *  to row ILST.
 *
 *  (A, B) must be in generalized real Schur canonical form (as returned
 *  by DGGES), i.e. A is block upper triangular with 1-by-1 and 2-by-2
 *  diagonal blocks. B is upper triangular.
 *
 *  Optionally, the matrices Q and Z of generalized Schur vectors are
 *  updated.
 *
 *         Q(in) * A(in) * Z(in)' = Q(out) * A(out) * Z(out)'
 *         Q(in) * B(in) * Z(in)' = Q(out) * B(out) * Z(out)'
 *
 *
 *  Arguments
 *  =========
 *
 *  WANTQ   (input) LOGICAL
 *          .TRUE. : update the left transformation matrix Q;
 *          .FALSE.: do not update Q.
 *
 *  WANTZ   (input) LOGICAL
 *          .TRUE. : update the right transformation matrix Z;
 *          .FALSE.: do not update Z.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the matrix A in generalized real Schur canonical
 *          form.
 *          On exit, the updated matrix A, again in generalized
 *          real Schur canonical form.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the matrix B in generalized real Schur canonical
 *          form (A,B).
 *          On exit, the updated matrix B, again in generalized
 *          real Schur canonical form (A,B).
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          On entry, if WANTQ = .TRUE., the orthogonal matrix Q.
 *          On exit, the updated matrix Q.
 *          If WANTQ = .FALSE., Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q. LDQ >= 1.
 *          If WANTQ = .TRUE., LDQ >= N.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ,N)
 *          On entry, if WANTZ = .TRUE., the orthogonal matrix Z.
 *          On exit, the updated matrix Z.
 *          If WANTZ = .FALSE., Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z. LDZ >= 1.
 *          If WANTZ = .TRUE., LDZ >= N.
 *
 *  IFST    (input/output) INTEGER
 *  ILST    (input/output) INTEGER
 *          Specify the reordering of the diagonal blocks of (A, B).
 *          The block with row index IFST is moved to row ILST, by a
 *          sequence of swapping between adjacent blocks.
 *          On exit, if IFST pointed on entry to the second row of
 *          a 2-by-2 block, it is changed to point to the first row;
 *          ILST always points to the first row of the block in its
 *          final position (which may differ from its input value by
 *          +1 or -1). 1 <= IFST, ILST <= N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          LWORK >= 1 when N <= 1, otherwise LWORK >= 4*N + 16.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *           =0:  successful exit.
 *           <0:  if INFO = -i, the i-th argument had an illegal value.
 *           =1:  The transformed matrix pair (A, B) would be too far
 *                from generalized Schur form; the problem is ill-
 *                conditioned. (A, B) may have been partially reordered,
 *                and ILST points to the first row of the current
 *                position of the block being moved.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Bo Kagstrom and Peter Poromaa, Department of Computing Science,
 *     Umea University, S-901 87 Umea, Sweden.
 *
 *  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the
 *      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in
 *      M.S. Moonen et al (eds), Linear Algebra for Large Scale and
 *      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTGEXC(int n, double* a, int lda, double* b, int ldb, double* q, int ldq, double* z, int ldz, int* ifst, int* ilst, double* work, int lwork)
{
    int info;
    ::F_DTGEXC(&n, a, &lda, b, &ldb, q, &ldq, z, &ldz, ifst, ilst, work, &lwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTGSEN reorders the generalized real Schur decomposition of a real
 *  matrix pair (A, B) (in terms of an orthonormal equivalence trans-
 *  formation Q' * (A, B) * Z), so that a selected cluster of eigenvalues
 *  appears in the leading diagonal blocks of the upper quasi-triangular
 *  matrix A and the upper triangular B. The leading columns of Q and
 *  Z form orthonormal bases of the corresponding left and right eigen-
 *  spaces (deflating subspaces). (A, B) must be in generalized real
 *  Schur canonical form (as returned by DGGES), i.e. A is block upper
 *  triangular with 1-by-1 and 2-by-2 diagonal blocks. B is upper
 *  triangular.
 *
 *  DTGSEN also computes the generalized eigenvalues
 *
 *              w(j) = (ALPHAR(j) + i*ALPHAI(j))/BETA(j)
 *
 *  of the reordered matrix pair (A, B).
 *
 *  Optionally, DTGSEN computes the estimates of reciprocal condition
 *  numbers for eigenvalues and eigenspaces. These are Difu[(A11,B11),
 *  (A22,B22)] and Difl[(A11,B11), (A22,B22)], i.e. the separation(s)
 *  between the matrix pairs (A11, B11) and (A22,B22) that correspond to
 *  the selected cluster and the eigenvalues outside the cluster, resp.,
 *  and norms of "projections" onto left and right eigenspaces w.r.t.
 *  the selected cluster in the (1,1)-block.
 *
 *  Arguments
 *  =========
 *
 *  IJOB    (input) INTEGER
 *          Specifies whether condition numbers are required for the
 *          cluster of eigenvalues (PL and PR) or the deflating subspaces
 *          (Difu and Difl):
 *           =0: Only reorder w.r.t. SELECT. No extras.
 *           =1: Reciprocal of norms of "projections" onto left and right
 *               eigenspaces w.r.t. the selected cluster (PL and PR).
 *           =2: Upper bounds on Difu and Difl. F-norm-based estimate
 *               (DIF(1:2)).
 *           =3: Estimate of Difu and Difl. 1-norm-based estimate
 *               (DIF(1:2)).
 *               About 5 times as expensive as IJOB = 2.
 *           =4: Compute PL, PR and DIF (i.e. 0, 1 and 2 above): Economic
 *               version to get it all.
 *           =5: Compute PL, PR and DIF (i.e. 0, 1 and 3 above)
 *
 *  WANTQ   (input) LOGICAL
 *          .TRUE. : update the left transformation matrix Q;
 *          .FALSE.: do not update Q.
 *
 *  WANTZ   (input) LOGICAL
 *          .TRUE. : update the right transformation matrix Z;
 *          .FALSE.: do not update Z.
 *
 *  SELECT  (input) LOGICAL array, dimension (N)
 *          SELECT specifies the eigenvalues in the selected cluster.
 *          To select a real eigenvalue w(j), SELECT(j) must be set to
 *          .TRUE.. To select a complex conjugate pair of eigenvalues
 *          w(j) and w(j+1), corresponding to a 2-by-2 diagonal block,
 *          either SELECT(j) or SELECT(j+1) or both must be set to
 *          .TRUE.; a complex conjugate pair of eigenvalues must be
 *          either both included in the cluster or both excluded.
 *
 *  N       (input) INTEGER
 *          The order of the matrices A and B. N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension(LDA,N)
 *          On entry, the upper quasi-triangular matrix A, with (A, B) in
 *          generalized real Schur canonical form.
 *          On exit, A is overwritten by the reordered matrix A.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension(LDB,N)
 *          On entry, the upper triangular matrix B, with (A, B) in
 *          generalized real Schur canonical form.
 *          On exit, B is overwritten by the reordered matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  ALPHAR  (output) DOUBLE PRECISION array, dimension (N)
 *  ALPHAI  (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, (ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N, will
 *          be the generalized eigenvalues.  ALPHAR(j) + ALPHAI(j)*i
 *          and BETA(j),j=1,...,N  are the diagonals of the complex Schur
 *          form (S,T) that would result if the 2-by-2 diagonal blocks of
 *          the real generalized Schur form of (A,B) were further reduced
 *          to triangular form using complex unitary transformations.
 *          If ALPHAI(j) is zero, then the j-th eigenvalue is real; if
 *          positive, then the j-th and (j+1)-st eigenvalues are a
 *          complex conjugate pair, with ALPHAI(j+1) negative.
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          On entry, if WANTQ = .TRUE., Q is an N-by-N matrix.
 *          On exit, Q has been postmultiplied by the left orthogonal
 *          transformation matrix which reorder (A, B); The leading M
 *          columns of Q form orthonormal bases for the specified pair of
 *          left eigenspaces (deflating subspaces).
 *          If WANTQ = .FALSE., Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.  LDQ >= 1;
 *          and if WANTQ = .TRUE., LDQ >= N.
 *
 *  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ,N)
 *          On entry, if WANTZ = .TRUE., Z is an N-by-N matrix.
 *          On exit, Z has been postmultiplied by the left orthogonal
 *          transformation matrix which reorder (A, B); The leading M
 *          columns of Z form orthonormal bases for the specified pair of
 *          left eigenspaces (deflating subspaces).
 *          If WANTZ = .FALSE., Z is not referenced.
 *
 *  LDZ     (input) INTEGER
 *          The leading dimension of the array Z. LDZ >= 1;
 *          If WANTZ = .TRUE., LDZ >= N.
 *
 *  M       (output) INTEGER
 *          The dimension of the specified pair of left and right eigen-
 *          spaces (deflating subspaces). 0 <= M <= N.
 *
 *  PL      (output) DOUBLE PRECISION
 *  PR      (output) DOUBLE PRECISION
 *          If IJOB = 1, 4 or 5, PL, PR are lower bounds on the
 *          reciprocal of the norm of "projections" onto left and right
 *          eigenspaces with respect to the selected cluster.
 *          0 < PL, PR <= 1.
 *          If M = 0 or M = N, PL = PR  = 1.
 *          If IJOB = 0, 2 or 3, PL and PR are not referenced.
 *
 *  DIF     (output) DOUBLE PRECISION array, dimension (2).
 *          If IJOB >= 2, DIF(1:2) store the estimates of Difu and Difl.
 *          If IJOB = 2 or 4, DIF(1:2) are F-norm-based upper bounds on
 *          Difu and Difl. If IJOB = 3 or 5, DIF(1:2) are 1-norm-based
 *          estimates of Difu and Difl.
 *          If M = 0 or N, DIF(1:2) = F-norm([A, B]).
 *          If IJOB = 0 or 1, DIF is not referenced.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array,
 *          dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >=  4*N+16.
 *          If IJOB = 1, 2 or 4, LWORK >= MAX(4*N+16, 2*M*(N-M)).
 *          If IJOB = 3 or 5, LWORK >= MAX(4*N+16, 4*M*(N-M)).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK. LIWORK >= 1.
 *          If IJOB = 1, 2 or 4, LIWORK >=  N+6.
 *          If IJOB = 3 or 5, LIWORK >= MAX(2*M*(N-M), N+6).
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal size of the IWORK array,
 *          returns this value as the first entry of the IWORK array, and
 *          no error message related to LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *            =0: Successful exit.
 *            <0: If INFO = -i, the i-th argument had an illegal value.
 *            =1: Reordering of (A, B) failed because the transformed
 *                matrix pair (A, B) would be too far from generalized
 *                Schur form; the problem is very ill-conditioned.
 *                (A, B) may have been partially reordered.
 *                If requested, 0 is returned in DIF(*), PL and PR.
 *
 *  Further Details
 *  ===============
 *
 *  DTGSEN first collects the selected eigenvalues by computing
 *  orthogonal U and W that move them to the top left corner of (A, B).
 *  In other words, the selected eigenvalues are the eigenvalues of
 *  (A11, B11) in:
 *
 *                U'*(A, B)*W = (A11 A12) (B11 B12) n1
 *                              ( 0  A22),( 0  B22) n2
 *                                n1  n2    n1  n2
 *
 *  where N = n1+n2 and U' means the transpose of U. The first n1 columns
 *  of U and W span the specified pair of left and right eigenspaces
 *  (deflating subspaces) of (A, B).
 *
 *  If (A, B) has been obtained from the generalized real Schur
 *  decomposition of a matrix pair (C, D) = Q*(A, B)*Z', then the
 *  reordered generalized real Schur form of (C, D) is given by
 *
 *           (C, D) = (Q*U)*(U'*(A, B)*W)*(Z*W)',
 *
 *  and the first n1 columns of Q*U and Z*W span the corresponding
 *  deflating subspaces of (C, D) (Q and Z store Q*U and Z*W, resp.).
 *
 *  Note that if the selected eigenvalue is sufficiently ill-conditioned,
 *  then its value may differ significantly from its value before
 *  reordering.
 *
 *  The reciprocal condition numbers of the left and right eigenspaces
 *  spanned by the first n1 columns of U and W (or Q*U and Z*W) may
 *  be returned in DIF(1:2), corresponding to Difu and Difl, resp.
 *
 *  The Difu and Difl are defined as:
 *
 *       Difu[(A11, B11), (A22, B22)] = sigma-min( Zu )
 *  and
 *       Difl[(A11, B11), (A22, B22)] = Difu[(A22, B22), (A11, B11)],
 *
 *  where sigma-min(Zu) is the smallest singular value of the
 *  (2*n1*n2)-by-(2*n1*n2) matrix
 *
 *       Zu = [ kron(In2, A11)  -kron(A22', In1) ]
 *            [ kron(In2, B11)  -kron(B22', In1) ].
 *
 *  Here, Inx is the identity matrix of size nx and A22' is the
 *  transpose of A22. kron(X, Y) is the Kronecker product between
 *  the matrices X and Y.
 *
 *  When DIF(2) is small, small changes in (A, B) can cause large changes
 *  in the deflating subspace. An approximate (asymptotic) bound on the
 *  maximum angular error in the computed deflating subspaces is
 *
 *       EPS * norm((A, B)) / DIF(2),
 *
 *  where EPS is the machine precision.
 *
 *  The reciprocal norm of the projectors on the left and right
 *  eigenspaces associated with (A11, B11) may be returned in PL and PR.
 *  They are computed as follows. First we compute L and R so that
 *  P*(A, B)*Q is block diagonal, where
 *
 *       P = ( I -L ) n1           Q = ( I R ) n1
 *           ( 0  I ) n2    and        ( 0 I ) n2
 *             n1 n2                    n1 n2
 *
 *  and (L, R) is the solution to the generalized Sylvester equation
 *
 *       A11*R - L*A22 = -A12
 *       B11*R - L*B22 = -B12
 *
 *  Then PL = (F-norm(L)**2+1)**(-1/2) and PR = (F-norm(R)**2+1)**(-1/2).
 *  An approximate (asymptotic) bound on the average absolute error of
 *  the selected eigenvalues is
 *
 *       EPS * norm((A, B)) / PL.
 *
 *  There are also global error bounds which valid for perturbations up
 *  to a certain restriction:  A lower bound (x) on the smallest
 *  F-norm(E,F) for which an eigenvalue of (A11, B11) may move and
 *  coalesce with an eigenvalue of (A22, B22) under perturbation (E,F),
 *  (i.e. (A + E, B + F), is
 *
 *   x = min(Difu,Difl)/((1/(PL*PL)+1/(PR*PR))**(1/2)+2*max(1/PL,1/PR)).
 *
 *  An approximate bound on x can be computed from DIF(1:2), PL and PR.
 *
 *  If y = ( F-norm(E,F) / x) <= 1, the angles between the perturbed
 *  (L', R') and unperturbed (L, R) left and right deflating subspaces
 *  associated with the selected cluster in the (1,1)-blocks can be
 *  bounded as
 *
 *   max-angle(L, L') <= arctan( y * PL / (1 - y * (1 - PL * PL)**(1/2))
 *   max-angle(R, R') <= arctan( y * PR / (1 - y * (1 - PR * PR)**(1/2))
 *
 *  See LAPACK User's Guide section 4.11 or the following references
 *  for more information.
 *
 *  Note that if the default method for computing the Frobenius-norm-
 *  based estimate DIF is not wanted (see DLATDF), then the parameter
 *  IDIFJB (see below) should be changed from 3 to 4 (routine DLATDF
 *  (IJOB = 2 will be used)). See DTGSYL for more details.
 *
 *  Based on contributions by
 *     Bo Kagstrom and Peter Poromaa, Department of Computing Science,
 *     Umea University, S-901 87 Umea, Sweden.
 *
 *  References
 *  ==========
 *
 *  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the
 *      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in
 *      M.S. Moonen et al (eds), Linear Algebra for Large Scale and
 *      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218.
 *
 *  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified
 *      Eigenvalues of a Regular Matrix Pair (A, B) and Condition
 *      Estimation: Theory, Algorithms and Software,
 *      Report UMINF - 94.04, Department of Computing Science, Umea
 *      University, S-901 87 Umea, Sweden, 1994. Also as LAPACK Working
 *      Note 87. To appear in Numerical Algorithms, 1996.
 *
 *  [3] B. Kagstrom and P. Poromaa, LAPACK-Style Algorithms and Software
 *      for Solving the Generalized Sylvester Equation and Estimating the
 *      Separation between Regular Matrix Pairs, Report UMINF - 93.23,
 *      Department of Computing Science, Umea University, S-901 87 Umea,
 *      Sweden, December 1993, Revised April 1994, Also as LAPACK Working
 *      Note 75. To appear in ACM Trans. on Math. Software, Vol 22, No 1,
 *      1996.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTGSEN(int ijob, int n, double* a, int lda, double* b, int ldb, double* alphar, double* alphai, double* beta, double* q, int ldq, double* z, int ldz, int* m, double* pl, double* pr, double* dif, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DTGSEN(&ijob, &n, a, &lda, b, &ldb, alphar, alphai, beta, q, &ldq, z, &ldz, m, pl, pr, dif, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTGSJA computes the generalized singular value decomposition (GSVD)
 *  of two real upper triangular (or trapezoidal) matrices A and B.
 *
 *  On entry, it is assumed that matrices A and B have the following
 *  forms, which may be obtained by the preprocessing subroutine DGGSVP
 *  from a general M-by-N matrix A and P-by-N matrix B:
 *
 *               N-K-L  K    L
 *     A =    K ( 0    A12  A13 ) if M-K-L >= 0;
 *            L ( 0     0   A23 )
 *        M-K-L ( 0     0    0  )
 *
 *             N-K-L  K    L
 *     A =  K ( 0    A12  A13 ) if M-K-L < 0;
 *        M-K ( 0     0   A23 )
 *
 *             N-K-L  K    L
 *     B =  L ( 0     0   B13 )
 *        P-L ( 0     0    0  )
 *
 *  where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
 *  upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0,
 *  otherwise A23 is (M-K)-by-L upper trapezoidal.
 *
 *  On exit,
 *
 *              U'*A*Q = D1*( 0 R ),    V'*B*Q = D2*( 0 R ),
 *
 *  where U, V and Q are orthogonal matrices, Z' denotes the transpose
 *  of Z, R is a nonsingular upper triangular matrix, and D1 and D2 are
 *  ``diagonal'' matrices, which are of the following structures:
 *
 *  If M-K-L >= 0,
 *
 *                      K  L
 *         D1 =     K ( I  0 )
 *                  L ( 0  C )
 *              M-K-L ( 0  0 )
 *
 *                    K  L
 *         D2 = L   ( 0  S )
 *              P-L ( 0  0 )
 *
 *                 N-K-L  K    L
 *    ( 0 R ) = K (  0   R11  R12 ) K
 *              L (  0    0   R22 ) L
 *
 *  where
 *
 *    C = diag( ALPHA(K+1), ... , ALPHA(K+L) ),
 *    S = diag( BETA(K+1),  ... , BETA(K+L) ),
 *    C**2 + S**2 = I.
 *
 *    R is stored in A(1:K+L,N-K-L+1:N) on exit.
 *
 *  If M-K-L < 0,
 *
 *                 K M-K K+L-M
 *      D1 =   K ( I  0    0   )
 *           M-K ( 0  C    0   )
 *
 *                   K M-K K+L-M
 *      D2 =   M-K ( 0  S    0   )
 *           K+L-M ( 0  0    I   )
 *             P-L ( 0  0    0   )
 *
 *                 N-K-L  K   M-K  K+L-M
 * ( 0 R ) =    K ( 0    R11  R12  R13  )
 *            M-K ( 0     0   R22  R23  )
 *          K+L-M ( 0     0    0   R33  )
 *
 *  where
 *  C = diag( ALPHA(K+1), ... , ALPHA(M) ),
 *  S = diag( BETA(K+1),  ... , BETA(M) ),
 *  C**2 + S**2 = I.
 *
 *  R = ( R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N) and R33 is stored
 *      (  0  R22 R23 )
 *  in B(M-K+1:L,N+M-K-L+1:N) on exit.
 *
 *  The computation of the orthogonal transformation matrices U, V or Q
 *  is optional.  These matrices may either be formed explicitly, or they
 *  may be postmultiplied into input matrices U1, V1, or Q1.
 *
 *  Arguments
 *  =========
 *
 *  JOBU    (input) CHARACTER*1
 *          = 'U':  U must contain an orthogonal matrix U1 on entry, and
 *                  the product U1*U is returned;
 *          = 'I':  U is initialized to the unit matrix, and the
 *                  orthogonal matrix U is returned;
 *          = 'N':  U is not computed.
 *
 *  JOBV    (input) CHARACTER*1
 *          = 'V':  V must contain an orthogonal matrix V1 on entry, and
 *                  the product V1*V is returned;
 *          = 'I':  V is initialized to the unit matrix, and the
 *                  orthogonal matrix V is returned;
 *          = 'N':  V is not computed.
 *
 *  JOBQ    (input) CHARACTER*1
 *          = 'Q':  Q must contain an orthogonal matrix Q1 on entry, and
 *                  the product Q1*Q is returned;
 *          = 'I':  Q is initialized to the unit matrix, and the
 *                  orthogonal matrix Q is returned;
 *          = 'N':  Q is not computed.
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  P       (input) INTEGER
 *          The number of rows of the matrix B.  P >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrices A and B.  N >= 0.
 *
 *  K       (input) INTEGER
 *  L       (input) INTEGER
 *          K and L specify the subblocks in the input matrices A and B:
 *          A23 = A(K+1:MIN(K+L,M),N-L+1:N) and B13 = B(1:L,N-L+1:N)
 *          of A and B, whose GSVD is going to be computed by DTGSJA.
 *          See Further Details.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the M-by-N matrix A.
 *          On exit, A(N-K+1:N,1:MIN(K+L,M) ) contains the triangular
 *          matrix R or part of R.  See Purpose for details.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N)
 *          On entry, the P-by-N matrix B.
 *          On exit, if necessary, B(M-K+1:L,N+M-K-L+1:N) contains
 *          a part of R.  See Purpose for details.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,P).
 *
 *  TOLA    (input) DOUBLE PRECISION
 *  TOLB    (input) DOUBLE PRECISION
 *          TOLA and TOLB are the convergence criteria for the Jacobi-
 *          Kogbetliantz iteration procedure. Generally, they are the
 *          same as used in the preprocessing step, say
 *              TOLA = max(M,N)*norm(A)*MAZHEPS,
 *              TOLB = max(P,N)*norm(B)*MAZHEPS.
 *
 *  ALPHA   (output) DOUBLE PRECISION array, dimension (N)
 *  BETA    (output) DOUBLE PRECISION array, dimension (N)
 *          On exit, ALPHA and BETA contain the generalized singular
 *          value pairs of A and B;
 *            ALPHA(1:K) = 1,
 *            BETA(1:K)  = 0,
 *          and if M-K-L >= 0,
 *            ALPHA(K+1:K+L) = diag(C),
 *            BETA(K+1:K+L)  = diag(S),
 *          or if M-K-L < 0,
 *            ALPHA(K+1:M)= C, ALPHA(M+1:K+L)= 0
 *            BETA(K+1:M) = S, BETA(M+1:K+L) = 1.
 *          Furthermore, if K+L < N,
 *            ALPHA(K+L+1:N) = 0 and
 *            BETA(K+L+1:N)  = 0.
 *
 *  U       (input/output) DOUBLE PRECISION array, dimension (LDU,M)
 *          On entry, if JOBU = 'U', U must contain a matrix U1 (usually
 *          the orthogonal matrix returned by DGGSVP).
 *          On exit,
 *          if JOBU = 'I', U contains the orthogonal matrix U;
 *          if JOBU = 'U', U contains the product U1*U.
 *          If JOBU = 'N', U is not referenced.
 *
 *  LDU     (input) INTEGER
 *          The leading dimension of the array U. LDU >= max(1,M) if
 *          JOBU = 'U'; LDU >= 1 otherwise.
 *
 *  V       (input/output) DOUBLE PRECISION array, dimension (LDV,P)
 *          On entry, if JOBV = 'V', V must contain a matrix V1 (usually
 *          the orthogonal matrix returned by DGGSVP).
 *          On exit,
 *          if JOBV = 'I', V contains the orthogonal matrix V;
 *          if JOBV = 'V', V contains the product V1*V.
 *          If JOBV = 'N', V is not referenced.
 *
 *  LDV     (input) INTEGER
 *          The leading dimension of the array V. LDV >= max(1,P) if
 *          JOBV = 'V'; LDV >= 1 otherwise.
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          On entry, if JOBQ = 'Q', Q must contain a matrix Q1 (usually
 *          the orthogonal matrix returned by DGGSVP).
 *          On exit,
 *          if JOBQ = 'I', Q contains the orthogonal matrix Q;
 *          if JOBQ = 'Q', Q contains the product Q1*Q.
 *          If JOBQ = 'N', Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q. LDQ >= max(1,N) if
 *          JOBQ = 'Q'; LDQ >= 1 otherwise.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
 *
 *  NCYCLE  (output) INTEGER
 *          The number of cycles required for convergence.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value.
 *          = 1:  the procedure does not converge after MAXIT cycles.
 *
 *  Internal Parameters
 *  ===================
 *
 *  MAXIT   INTEGER
 *          MAXIT specifies the total loops that the iterative procedure
 *          may take. If after MAXIT cycles, the routine fails to
 *          converge, we return INFO = 1.
 *
 *  Further Details
 *  ===============
 *
 *  DTGSJA essentially uses a variant of Kogbetliantz algorithm to reduce
 *  min(L,M-K)-by-L triangular (or trapezoidal) matrix A23 and L-by-L
 *  matrix B13 to the form:
 *
 *           U1'*A13*Q1 = C1*R1; V1'*B13*Q1 = S1*R1,
 *
 *  where U1, V1 and Q1 are orthogonal matrix, and Z' is the transpose
 *  of Z.  C1 and S1 are diagonal matrices satisfying
 *
 *                C1**2 + S1**2 = I,
 *
 *  and R1 is an L-by-L nonsingular upper triangular matrix.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTGSJA(char jobu, char jobv, char jobq, int m, int p, int n, int k, int l, double* a, int lda, double* b, int ldb, double tola, double tolb, double* alpha, double* beta, double* u, int ldu, double* v, int ldv, double* q, int ldq, double* work, int* ncycle)
{
    int info;
    ::F_DTGSJA(&jobu, &jobv, &jobq, &m, &p, &n, &k, &l, a, &lda, b, &ldb, &tola, &tolb, alpha, beta, u, &ldu, v, &ldv, q, &ldq, work, ncycle, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTGSNA estimates reciprocal condition numbers for specified
 *  eigenvalues and/or eigenvectors of a matrix pair (A, B) in
 *  generalized real Schur canonical form (or of any matrix pair
 *  (Q*A*Z', Q*B*Z') with orthogonal matrices Q and Z, where
 *  Z' denotes the transpose of Z.
 *
 *  (A, B) must be in generalized real Schur form (as returned by DGGES),
 *  i.e. A is block upper triangular with 1-by-1 and 2-by-2 diagonal
 *  blocks. B is upper triangular.
 *
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies whether condition numbers are required for
 *          eigenvalues (S) or eigenvectors (DIF):
 *          = 'E': for eigenvalues only (S);
 *          = 'V': for eigenvectors only (DIF);
 *          = 'B': for both eigenvalues and eigenvectors (S and DIF).
 *
 *  HOWMNY  (input) CHARACTER*1
 *          = 'A': compute condition numbers for all eigenpairs;
 *          = 'S': compute condition numbers for selected eigenpairs
 *                 specified by the array SELECT.
 *
 *  SELECT  (input) LOGICAL array, dimension (N)
 *          If HOWMNY = 'S', SELECT specifies the eigenpairs for which
 *          condition numbers are required. To select condition numbers
 *          for the eigenpair corresponding to a real eigenvalue w(j),
 *          SELECT(j) must be set to .TRUE.. To select condition numbers
 *          corresponding to a complex conjugate pair of eigenvalues w(j)
 *          and w(j+1), either SELECT(j) or SELECT(j+1) or both, must be
 *          set to .TRUE..
 *          If HOWMNY = 'A', SELECT is not referenced.
 *
 *  N       (input) INTEGER
 *          The order of the square matrix pair (A, B). N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The upper quasi-triangular matrix A in the pair (A,B).
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,N).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,N)
 *          The upper triangular matrix B in the pair (A,B).
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION array, dimension (LDVL,M)
 *          If JOB = 'E' or 'B', VL must contain left eigenvectors of
 *          (A, B), corresponding to the eigenpairs specified by HOWMNY
 *          and SELECT. The eigenvectors must be stored in consecutive
 *          columns of VL, as returned by DTGEVC.
 *          If JOB = 'V', VL is not referenced.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the array VL. LDVL >= 1.
 *          If JOB = 'E' or 'B', LDVL >= N.
 *
 *  VR      (input) DOUBLE PRECISION array, dimension (LDVR,M)
 *          If JOB = 'E' or 'B', VR must contain right eigenvectors of
 *          (A, B), corresponding to the eigenpairs specified by HOWMNY
 *          and SELECT. The eigenvectors must be stored in consecutive
 *          columns ov VR, as returned by DTGEVC.
 *          If JOB = 'V', VR is not referenced.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR. LDVR >= 1.
 *          If JOB = 'E' or 'B', LDVR >= N.
 *
 *  S       (output) DOUBLE PRECISION array, dimension (MM)
 *          If JOB = 'E' or 'B', the reciprocal condition numbers of the
 *          selected eigenvalues, stored in consecutive elements of the
 *          array. For a complex conjugate pair of eigenvalues two
 *          consecutive elements of S are set to the same value. Thus
 *          S(j), DIF(j), and the j-th columns of VL and VR all
 *          correspond to the same eigenpair (but not in general the
 *          j-th eigenpair, unless all eigenpairs are selected).
 *          If JOB = 'V', S is not referenced.
 *
 *  DIF     (output) DOUBLE PRECISION array, dimension (MM)
 *          If JOB = 'V' or 'B', the estimated reciprocal condition
 *          numbers of the selected eigenvectors, stored in consecutive
 *          elements of the array. For a complex eigenvector two
 *          consecutive elements of DIF are set to the same value. If
 *          the eigenvalues cannot be reordered to compute DIF(j), DIF(j)
 *          is set to 0; this can only occur when the true value would be
 *          very small anyway.
 *          If JOB = 'E', DIF is not referenced.
 *
 *  MM      (input) INTEGER
 *          The number of elements in the arrays S and DIF. MM >= M.
 *
 *  M       (output) INTEGER
 *          The number of elements of the arrays S and DIF used to store
 *          the specified condition numbers; for each selected real
 *          eigenvalue one element is used, and for each selected complex
 *          conjugate pair of eigenvalues, two elements are used.
 *          If HOWMNY = 'A', M is set to N.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK >= max(1,N).
 *          If JOB = 'V' or 'B' LWORK >= 2*N*(N+2)+16.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (N + 6)
 *          If JOB = 'E', IWORK is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          =0: Successful exit
 *          <0: If INFO = -i, the i-th argument had an illegal value
 *
 *
 *  Further Details
 *  ===============
 *
 *  The reciprocal of the condition number of a generalized eigenvalue
 *  w = (a, b) is defined as
 *
 *       S(w) = (|u'Av|**2 + |u'Bv|**2)**(1/2) / (norm(u)*norm(v))
 *
 *  where u and v are the left and right eigenvectors of (A, B)
 *  corresponding to w; |z| denotes the absolute value of the complex
 *  number, and norm(u) denotes the 2-norm of the vector u.
 *  The pair (a, b) corresponds to an eigenvalue w = a/b (= u'Av/u'Bv)
 *  of the matrix pair (A, B). If both a and b equal zero, then (A B) is
 *  singular and S(I) = -1 is returned.
 *
 *  An approximate error bound on the chordal distance between the i-th
 *  computed generalized eigenvalue w and the corresponding exact
 *  eigenvalue lambda is
 *
 *       chord(w, lambda) <= EPS * norm(A, B) / S(I)
 *
 *  where EPS is the machine precision.
 *
 *  The reciprocal of the condition number DIF(i) of right eigenvector u
 *  and left eigenvector v corresponding to the generalized eigenvalue w
 *  is defined as follows:
 *
 *  a) If the i-th eigenvalue w = (a,b) is real
 *
 *     Suppose U and V are orthogonal transformations such that
 *
 *                U'*(A, B)*V  = (S, T) = ( a   *  ) ( b  *  )  1
 *                                        ( 0  S22 ),( 0 T22 )  n-1
 *                                          1  n-1     1 n-1
 *
 *     Then the reciprocal condition number DIF(i) is
 *
 *                Difl((a, b), (S22, T22)) = sigma-min( Zl ),
 *
 *     where sigma-min(Zl) denotes the smallest singular value of the
 *     2(n-1)-by-2(n-1) matrix
 *
 *         Zl = [ kron(a, In-1)  -kron(1, S22) ]
 *              [ kron(b, In-1)  -kron(1, T22) ] .
 *
 *     Here In-1 is the identity matrix of size n-1. kron(X, Y) is the
 *     Kronecker product between the matrices X and Y.
 *
 *     Note that if the default method for computing DIF(i) is wanted
 *     (see DLATDF), then the parameter DIFDRI (see below) should be
 *     changed from 3 to 4 (routine DLATDF(IJOB = 2 will be used)).
 *     See DTGSYL for more details.
 *
 *  b) If the i-th and (i+1)-th eigenvalues are complex conjugate pair,
 *
 *     Suppose U and V are orthogonal transformations such that
 *
 *                U'*(A, B)*V = (S, T) = ( S11  *   ) ( T11  *  )  2
 *                                       ( 0    S22 ),( 0    T22) n-2
 *                                         2    n-2     2    n-2
 *
 *     and (S11, T11) corresponds to the complex conjugate eigenvalue
 *     pair (w, conjg(w)). There exist unitary matrices U1 and V1 such
 *     that
 *
 *         U1'*S11*V1 = ( s11 s12 )   and U1'*T11*V1 = ( t11 t12 )
 *                      (  0  s22 )                    (  0  t22 )
 *
 *     where the generalized eigenvalues w = s11/t11 and
 *     conjg(w) = s22/t22.
 *
 *     Then the reciprocal condition number DIF(i) is bounded by
 *
 *         min( d1, max( 1, |real(s11)/real(s22)| )*d2 )
 *
 *     where, d1 = Difl((s11, t11), (s22, t22)) = sigma-min(Z1), where
 *     Z1 is the complex 2-by-2 matrix
 *
 *              Z1 =  [ s11  -s22 ]
 *                    [ t11  -t22 ],
 *
 *     This is done by computing (using real arithmetic) the
 *     roots of the characteristical polynomial det(Z1' * Z1 - lambda I),
 *     where Z1' denotes the conjugate transpose of Z1 and det(X) denotes
 *     the determinant of X.
 *
 *     and d2 is an upper bound on Difl((S11, T11), (S22, T22)), i.e. an
 *     upper bound on sigma-min(Z2), where Z2 is (2n-2)-by-(2n-2)
 *
 *              Z2 = [ kron(S11', In-2)  -kron(I2, S22) ]
 *                   [ kron(T11', In-2)  -kron(I2, T22) ]
 *
 *     Note that if the default method for computing DIF is wanted (see
 *     DLATDF), then the parameter DIFDRI (see below) should be changed
 *     from 3 to 4 (routine DLATDF(IJOB = 2 will be used)). See DTGSYL
 *     for more details.
 *
 *  For each eigenvalue/vector specified by SELECT, DIF stores a
 *  Frobenius norm-based estimate of Difl.
 *
 *  An approximate error bound for the i-th computed eigenvector VL(i) or
 *  VR(i) is given by
 *
 *             EPS * norm(A, B) / DIF(i).
 *
 *  See ref. [2-3] for more details and further references.
 *
 *  Based on contributions by
 *     Bo Kagstrom and Peter Poromaa, Department of Computing Science,
 *     Umea University, S-901 87 Umea, Sweden.
 *
 *  References
 *  ==========
 *
 *  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the
 *      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in
 *      M.S. Moonen et al (eds), Linear Algebra for Large Scale and
 *      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218.
 *
 *  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified
 *      Eigenvalues of a Regular Matrix Pair (A, B) and Condition
 *      Estimation: Theory, Algorithms and Software,
 *      Report UMINF - 94.04, Department of Computing Science, Umea
 *      University, S-901 87 Umea, Sweden, 1994. Also as LAPACK Working
 *      Note 87. To appear in Numerical Algorithms, 1996.
 *
 *  [3] B. Kagstrom and P. Poromaa, LAPACK-Style Algorithms and Software
 *      for Solving the Generalized Sylvester Equation and Estimating the
 *      Separation between Regular Matrix Pairs, Report UMINF - 93.23,
 *      Department of Computing Science, Umea University, S-901 87 Umea,
 *      Sweden, December 1993, Revised April 1994, Also as LAPACK Working
 *      Note 75.  To appear in ACM Trans. on Math. Software, Vol 22,
 *      No 1, 1996.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTGSNA(char job, char howmny, int n, double* a, int lda, double* b, int ldb, double* vl, int ldvl, double* vr, int ldvr, double* s, double* dif, int mm, int* m, double* work, int lwork, int* iwork)
{
    int info;
    ::F_DTGSNA(&job, &howmny, &n, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr, s, dif, &mm, m, work, &lwork, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTGSYL solves the generalized Sylvester equation:
 *
 *              A * R - L * B = scale * C                 (1)
 *              D * R - L * E = scale * F
 *
 *  where R and L are unknown m-by-n matrices, (A, D), (B, E) and
 *  (C, F) are given matrix pairs of size m-by-m, n-by-n and m-by-n,
 *  respectively, with real entries. (A, D) and (B, E) must be in
 *  generalized (real) Schur canonical form, i.e. A, B are upper quasi
 *  triangular and D, E are upper triangular.
 *
 *  The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output
 *  scaling factor chosen to avoid overflow.
 *
 *  In matrix notation (1) is equivalent to solve  Zx = scale b, where
 *  Z is defined as
 *
 *             Z = [ kron(In, A)  -kron(B', Im) ]         (2)
 *                 [ kron(In, D)  -kron(E', Im) ].
 *
 *  Here Ik is the identity matrix of size k and X' is the transpose of
 *  X. kron(X, Y) is the Kronecker product between the matrices X and Y.
 *
 *  If TRANS = 'T', DTGSYL solves the transposed system Z'*y = scale*b,
 *  which is equivalent to solve for R and L in
 *
 *              A' * R  + D' * L   = scale *  C           (3)
 *              R  * B' + L  * E'  = scale * (-F)
 *
 *  This case (TRANS = 'T') is used to compute an one-norm-based estimate
 *  of Dif[(A,D), (B,E)], the separation between the matrix pairs (A,D)
 *  and (B,E), using DLACON.
 *
 *  If IJOB >= 1, DTGSYL computes a Frobenius norm-based estimate
 *  of Dif[(A,D),(B,E)]. That is, the reciprocal of a lower bound on the
 *  reciprocal of the smallest singular value of Z. See [1-2] for more
 *  information.
 *
 *  This is a level 3 BLAS algorithm.
 *
 *  Arguments
 *  =========
 *
 *  TRANS   (input) CHARACTER*1
 *          = 'N', solve the generalized Sylvester equation (1).
 *          = 'T', solve the 'transposed' system (3).
 *
 *  IJOB    (input) INTEGER
 *          Specifies what kind of functionality to be performed.
 *           =0: solve (1) only.
 *           =1: The functionality of 0 and 3.
 *           =2: The functionality of 0 and 4.
 *           =3: Only an estimate of Dif[(A,D), (B,E)] is computed.
 *               (look ahead strategy IJOB  = 1 is used).
 *           =4: Only an estimate of Dif[(A,D), (B,E)] is computed.
 *               ( DGECON on sub-systems is used ).
 *          Not referenced if TRANS = 'T'.
 *
 *  M       (input) INTEGER
 *          The order of the matrices A and D, and the row dimension of
 *          the matrices C, F, R and L.
 *
 *  N       (input) INTEGER
 *          The order of the matrices B and E, and the column dimension
 *          of the matrices C, F, R and L.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA, M)
 *          The upper quasi triangular matrix A.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1, M).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB, N)
 *          The upper quasi triangular matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1, N).
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC, N)
 *          On entry, C contains the right-hand-side of the first matrix
 *          equation in (1) or (3).
 *          On exit, if IJOB = 0, 1 or 2, C has been overwritten by
 *          the solution R. If IJOB = 3 or 4 and TRANS = 'N', C holds R,
 *          the solution achieved during the computation of the
 *          Dif-estimate.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1, M).
 *
 *  D       (input) DOUBLE PRECISION array, dimension (LDD, M)
 *          The upper triangular matrix D.
 *
 *  LDD     (input) INTEGER
 *          The leading dimension of the array D. LDD >= max(1, M).
 *
 *  E       (input) DOUBLE PRECISION array, dimension (LDE, N)
 *          The upper triangular matrix E.
 *
 *  LDE     (input) INTEGER
 *          The leading dimension of the array E. LDE >= max(1, N).
 *
 *  F       (input/output) DOUBLE PRECISION array, dimension (LDF, N)
 *          On entry, F contains the right-hand-side of the second matrix
 *          equation in (1) or (3).
 *          On exit, if IJOB = 0, 1 or 2, F has been overwritten by
 *          the solution L. If IJOB = 3 or 4 and TRANS = 'N', F holds L,
 *          the solution achieved during the computation of the
 *          Dif-estimate.
 *
 *  LDF     (input) INTEGER
 *          The leading dimension of the array F. LDF >= max(1, M).
 *
 *  DIF     (output) DOUBLE PRECISION
 *          On exit DIF is the reciprocal of a lower bound of the
 *          reciprocal of the Dif-function, i.e. DIF is an upper bound of
 *          Dif[(A,D), (B,E)] = sigma_min(Z), where Z as in (2).
 *          IF IJOB = 0 or TRANS = 'T', DIF is not touched.
 *
 *  SCALE   (output) DOUBLE PRECISION
 *          On exit SCALE is the scaling factor in (1) or (3).
 *          If 0 < SCALE < 1, C and F hold the solutions R and L, resp.,
 *          to a slightly perturbed system but the input matrices A, B, D
 *          and E have not been changed. If SCALE = 0, C and F hold the
 *          solutions R and L, respectively, to the homogeneous system
 *          with C = F = 0. Normally, SCALE = 1.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK. LWORK > = 1.
 *          If IJOB = 1 or 2 and TRANS = 'N', LWORK >= max(1,2*M*N).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (M+N+6)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *            =0: successful exit
 *            <0: If INFO = -i, the i-th argument had an illegal value.
 *            >0: (A, D) and (B, E) have common or close eigenvalues.
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *     Bo Kagstrom and Peter Poromaa, Department of Computing Science,
 *     Umea University, S-901 87 Umea, Sweden.
 *
 *  [1] B. Kagstrom and P. Poromaa, LAPACK-Style Algorithms and Software
 *      for Solving the Generalized Sylvester Equation and Estimating the
 *      Separation between Regular Matrix Pairs, Report UMINF - 93.23,
 *      Department of Computing Science, Umea University, S-901 87 Umea,
 *      Sweden, December 1993, Revised April 1994, Also as LAPACK Working
 *      Note 75.  To appear in ACM Trans. on Math. Software, Vol 22,
 *      No 1, 1996.
 *
 *  [2] B. Kagstrom, A Perturbation Analysis of the Generalized Sylvester
 *      Equation (AR - LB, DR - LE ) = (C, F), SIAM J. Matrix Anal.
 *      Appl., 15(4):1045-1060, 1994
 *
 *  [3] B. Kagstrom and L. Westin, Generalized Schur Methods with
 *      Condition Estimators for Solving the Generalized Sylvester
 *      Equation, IEEE Transactions on Automatic Control, Vol. 34, No. 7,
 *      July 1989, pp 745-751.
 *
 *  =====================================================================
 *  Replaced various illegal calls to DCOPY by calls to DLASET.
 *  Sven Hammarling, 1/5/02.
 *
 *     .. Parameters ..
 **/
int C_DTGSYL(char trans, int ijob, int m, int n, double* a, int lda, double* b, int ldb, double* c, int ldc, double* d, int ldd, double* e, int lde, double* f, int ldf, double* dif, double* scale, double* work, int lwork, int* iwork)
{
    int info;
    ::F_DTGSYL(&trans, &ijob, &m, &n, a, &lda, b, &ldb, c, &ldc, d, &ldd, e, &lde, f, &ldf, dif, scale, work, &lwork, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTPCON estimates the reciprocal of the condition number of a packed
 *  triangular matrix A, in either the 1-norm or the infinity-norm.
 *
 *  The norm of A is computed and an estimate is obtained for
 *  norm(inv(A)), then the reciprocal of the condition number is
 *  computed as
 *     RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 *  Arguments
 *  =========
 *
 *  NORM    (input) CHARACTER*1
 *          Specifies whether the 1-norm condition number or the
 *          infinity-norm condition number is required:
 *          = '1' or 'O':  1-norm;
 *          = 'I':         Infinity-norm.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangular matrix A, packed columnwise in
 *          a linear array.  The j-th column of A is stored in the array
 *          AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *          If DIAG = 'U', the diagonal elements of A are not referenced
 *          and are assumed to be 1.
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(norm(A) * norm(inv(A))).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTPCON(char norm, char uplo, char diag, int n, double* ap, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DTPCON(&norm, &uplo, &diag, &n, ap, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTPRFS provides error bounds and backward error estimates for the
 *  solution to a system of linear equations with a triangular packed
 *  coefficient matrix.
 *
 *  The solution matrix X must be computed by DTPTRS or some other
 *  means before entering this routine.  DTPRFS does not do iterative
 *  refinement because doing so cannot improve the backward error.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangular matrix A, packed columnwise in
 *          a linear array.  The j-th column of A is stored in the array
 *          AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *          If DIAG = 'U', the diagonal elements of A are not referenced
 *          and are assumed to be 1.
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          The solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTPRFS(char uplo, char trans, char diag, int n, int nrhs, double* ap, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DTPRFS(&uplo, &trans, &diag, &n, &nrhs, ap, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTPTRI computes the inverse of a real upper or lower triangular
 *  matrix A stored in packed format.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          On entry, the upper or lower triangular matrix A, stored
 *          columnwise in a linear array.  The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*((2*n-j)/2) = A(i,j) for j<=i<=n.
 *          See below for further details.
 *          On exit, the (triangular) inverse of the original matrix, in
 *          the same packed storage format.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, A(i,i) is exactly zero.  The triangular
 *                matrix is singular and its inverse can not be computed.
 *
 *  Further Details
 *  ===============
 *
 *  A triangular matrix A can be transferred to packed storage using one
 *  of the following program segments:
 *
 *  UPLO = 'U':                      UPLO = 'L':
 *
 *        JC = 1                           JC = 1
 *        DO 2 J = 1, N                    DO 2 J = 1, N
 *           DO 1 I = 1, J                    DO 1 I = J, N
 *              AP(JC+I-1) = A(I,J)              AP(JC+I-J) = A(I,J)
 *      1    CONTINUE                    1    CONTINUE
 *           JC = JC + J                      JC = JC + N - J + 1
 *      2 CONTINUE                       2 CONTINUE
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTPTRI(char uplo, char diag, int n, double* ap)
{
    int info;
    ::F_DTPTRI(&uplo, &diag, &n, ap, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTPTRS solves a triangular system of the form
 *
 *     A * X = B  or  A**T * X = B,
 *
 *  where A is a triangular matrix of order N stored in packed format,
 *  and B is an N-by-NRHS matrix.  A check is made to verify that A is
 *  nonsingular.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2)
 *          The upper or lower triangular matrix A, packed columnwise in
 *          a linear array.  The j-th column of A is stored in the array
 *          AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n.
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, if INFO = 0, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          > 0:  if INFO = i, the i-th diagonal element of A is zero,
 *                indicating that the matrix is singular and the
 *                solutions X have not been computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTPTRS(char uplo, char trans, char diag, int n, int nrhs, double* ap, double* b, int ldb)
{
    int info;
    ::F_DTPTRS(&uplo, &trans, &diag, &n, &nrhs, ap, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRCON estimates the reciprocal of the condition number of a
 *  triangular matrix A, in either the 1-norm or the infinity-norm.
 *
 *  The norm of A is computed and an estimate is obtained for
 *  norm(inv(A)), then the reciprocal of the condition number is
 *  computed as
 *     RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 *  Arguments
 *  =========
 *
 *  NORM    (input) CHARACTER*1
 *          Specifies whether the 1-norm condition number or the
 *          infinity-norm condition number is required:
 *          = '1' or 'O':  1-norm;
 *          = 'I':         Infinity-norm.
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The triangular matrix A.  If UPLO = 'U', the leading N-by-N
 *          upper triangular part of the array A contains the upper
 *          triangular matrix, and the strictly lower triangular part of
 *          A is not referenced.  If UPLO = 'L', the leading N-by-N lower
 *          triangular part of the array A contains the lower triangular
 *          matrix, and the strictly upper triangular part of A is not
 *          referenced.  If DIAG = 'U', the diagonal elements of A are
 *          also not referenced and are assumed to be 1.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  RCOND   (output) DOUBLE PRECISION
 *          The reciprocal of the condition number of the matrix A,
 *          computed as RCOND = 1/(norm(A) * norm(inv(A))).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRCON(char norm, char uplo, char diag, int n, double* a, int lda, double* rcond, double* work, int* iwork)
{
    int info;
    ::F_DTRCON(&norm, &uplo, &diag, &n, a, &lda, rcond, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTREVC computes some or all of the right and/or left eigenvectors of
 *  a real upper quasi-triangular matrix T.
 *  Matrices of this type are produced by the Schur factorization of
 *  a real general matrix:  A = Q*T*Q**T, as computed by DHSEQR.
 *
 *  The right eigenvector x and the left eigenvector y of T corresponding
 *  to an eigenvalue w are defined by:
 *
 *     T*x = w*x,     (y**H)*T = w*(y**H)
 *
 *  where y**H denotes the conjugate transpose of y.
 *  The eigenvalues are not input to this routine, but are read directly
 *  from the diagonal blocks of T.
 *
 *  This routine returns the matrices X and/or Y of right and left
 *  eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
 *  input matrix.  If Q is the orthogonal factor that reduces a matrix
 *  A to Schur form T, then Q*X and Q*Y are the matrices of right and
 *  left eigenvectors of A.
 *
 *  Arguments
 *  =========
 *
 *  SIDE    (input) CHARACTER*1
 *          = 'R':  compute right eigenvectors only;
 *          = 'L':  compute left eigenvectors only;
 *          = 'B':  compute both right and left eigenvectors.
 *
 *  HOWMNY  (input) CHARACTER*1
 *          = 'A':  compute all right and/or left eigenvectors;
 *          = 'B':  compute all right and/or left eigenvectors,
 *                  backtransformed by the matrices in VR and/or VL;
 *          = 'S':  compute selected right and/or left eigenvectors,
 *                  as indicated by the logical array SELECT.
 *
 *  SELECT  (input/output) LOGICAL array, dimension (N)
 *          If HOWMNY = 'S', SELECT specifies the eigenvectors to be
 *          computed.
 *          If w(j) is a real eigenvalue, the corresponding real
 *          eigenvector is computed if SELECT(j) is .TRUE..
 *          If w(j) and w(j+1) are the real and imaginary parts of a
 *          complex eigenvalue, the corresponding complex eigenvector is
 *          computed if either SELECT(j) or SELECT(j+1) is .TRUE., and
 *          on exit SELECT(j) is set to .TRUE. and SELECT(j+1) is set to
 *          .FALSE..
 *          Not referenced if HOWMNY = 'A' or 'B'.
 *
 *  N       (input) INTEGER
 *          The order of the matrix T. N >= 0.
 *
 *  T       (input) DOUBLE PRECISION array, dimension (LDT,N)
 *          The upper quasi-triangular matrix T in Schur canonical form.
 *
 *  LDT     (input) INTEGER
 *          The leading dimension of the array T. LDT >= max(1,N).
 *
 *  VL      (input/output) DOUBLE PRECISION array, dimension (LDVL,MM)
 *          On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must
 *          contain an N-by-N matrix Q (usually the orthogonal matrix Q
 *          of Schur vectors returned by DHSEQR).
 *          On exit, if SIDE = 'L' or 'B', VL contains:
 *          if HOWMNY = 'A', the matrix Y of left eigenvectors of T;
 *          if HOWMNY = 'B', the matrix Q*Y;
 *          if HOWMNY = 'S', the left eigenvectors of T specified by
 *                           SELECT, stored consecutively in the columns
 *                           of VL, in the same order as their
 *                           eigenvalues.
 *          A complex eigenvector corresponding to a complex eigenvalue
 *          is stored in two consecutive columns, the first holding the
 *          real part, and the second the imaginary part.
 *          Not referenced if SIDE = 'R'.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the array VL.  LDVL >= 1, and if
 *          SIDE = 'L' or 'B', LDVL >= N.
 *
 *  VR      (input/output) DOUBLE PRECISION array, dimension (LDVR,MM)
 *          On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must
 *          contain an N-by-N matrix Q (usually the orthogonal matrix Q
 *          of Schur vectors returned by DHSEQR).
 *          On exit, if SIDE = 'R' or 'B', VR contains:
 *          if HOWMNY = 'A', the matrix X of right eigenvectors of T;
 *          if HOWMNY = 'B', the matrix Q*X;
 *          if HOWMNY = 'S', the right eigenvectors of T specified by
 *                           SELECT, stored consecutively in the columns
 *                           of VR, in the same order as their
 *                           eigenvalues.
 *          A complex eigenvector corresponding to a complex eigenvalue
 *          is stored in two consecutive columns, the first holding the
 *          real part and the second the imaginary part.
 *          Not referenced if SIDE = 'L'.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR.  LDVR >= 1, and if
 *          SIDE = 'R' or 'B', LDVR >= N.
 *
 *  MM      (input) INTEGER
 *          The number of columns in the arrays VL and/or VR. MM >= M.
 *
 *  M       (output) INTEGER
 *          The number of columns in the arrays VL and/or VR actually
 *          used to store the eigenvectors.
 *          If HOWMNY = 'A' or 'B', M is set to N.
 *          Each selected real eigenvector occupies one column and each
 *          selected complex eigenvector occupies two columns.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The algorithm used in this program is basically backward (forward)
 *  substitution, with scaling to make the the code robust against
 *  possible overflow.
 *
 *  Each eigenvector is normalized so that the element of largest
 *  magnitude has magnitude 1; here the magnitude of a complex number
 *  (x,y) is taken to be |x| + |y|.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTREVC(char side, char howmny, int n, double* t, int ldt, double* vl, int ldvl, double* vr, int ldvr, int mm, int* m, double* work)
{
    int info;
    ::F_DTREVC(&side, &howmny, &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, m, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTREXC reorders the real Schur factorization of a real matrix
 *  A = Q*T*Q**T, so that the diagonal block of T with row index IFST is
 *  moved to row ILST.
 *
 *  The real Schur form T is reordered by an orthogonal similarity
 *  transformation Z**T*T*Z, and optionally the matrix Q of Schur vectors
 *  is updated by postmultiplying it with Z.
 *
 *  T must be in Schur canonical form (as returned by DHSEQR), that is,
 *  block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 *  2-by-2 diagonal block has its diagonal elements equal and its
 *  off-diagonal elements of opposite sign.
 *
 *  Arguments
 *  =========
 *
 *  COMPQ   (input) CHARACTER*1
 *          = 'V':  update the matrix Q of Schur vectors;
 *          = 'N':  do not update Q.
 *
 *  N       (input) INTEGER
 *          The order of the matrix T. N >= 0.
 *
 *  T       (input/output) DOUBLE PRECISION array, dimension (LDT,N)
 *          On entry, the upper quasi-triangular matrix T, in Schur
 *          Schur canonical form.
 *          On exit, the reordered upper quasi-triangular matrix, again
 *          in Schur canonical form.
 *
 *  LDT     (input) INTEGER
 *          The leading dimension of the array T. LDT >= max(1,N).
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          On entry, if COMPQ = 'V', the matrix Q of Schur vectors.
 *          On exit, if COMPQ = 'V', Q has been postmultiplied by the
 *          orthogonal transformation matrix Z which reorders T.
 *          If COMPQ = 'N', Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.  LDQ >= max(1,N).
 *
 *  IFST    (input/output) INTEGER
 *  ILST    (input/output) INTEGER
 *          Specify the reordering of the diagonal blocks of T.
 *          The block with row index IFST is moved to row ILST, by a
 *          sequence of transpositions between adjacent blocks.
 *          On exit, if IFST pointed on entry to the second row of a
 *          2-by-2 block, it is changed to point to the first row; ILST
 *          always points to the first row of the block in its final
 *          position (which may differ from its input value by +1 or -1).
 *          1 <= IFST <= N; 1 <= ILST <= N.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *          = 1:  two adjacent blocks were too close to swap (the problem
 *                is very ill-conditioned); T may have been partially
 *                reordered, and ILST points to the first row of the
 *                current position of the block being moved.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTREXC(char compq, int n, double* t, int ldt, double* q, int ldq, int* ifst, int* ilst, double* work)
{
    int info;
    ::F_DTREXC(&compq, &n, t, &ldt, q, &ldq, ifst, ilst, work, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRRFS provides error bounds and backward error estimates for the
 *  solution to a system of linear equations with a triangular
 *  coefficient matrix.
 *
 *  The solution matrix X must be computed by DTRTRS or some other
 *  means before entering this routine.  DTRRFS does not do iterative
 *  refinement because doing so cannot improve the backward error.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrices B and X.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The triangular matrix A.  If UPLO = 'U', the leading N-by-N
 *          upper triangular part of the array A contains the upper
 *          triangular matrix, and the strictly lower triangular part of
 *          A is not referenced.  If UPLO = 'L', the leading N-by-N lower
 *          triangular part of the array A contains the lower triangular
 *          matrix, and the strictly upper triangular part of A is not
 *          referenced.  If DIAG = 'U', the diagonal elements of A are
 *          also not referenced and are assumed to be 1.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          The right hand side matrix B.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  X       (input) DOUBLE PRECISION array, dimension (LDX,NRHS)
 *          The solution matrix X.
 *
 *  LDX     (input) INTEGER
 *          The leading dimension of the array X.  LDX >= max(1,N).
 *
 *  FERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The estimated forward error bound for each solution vector
 *          X(j) (the j-th column of the solution matrix X).
 *          If XTRUE is the true solution corresponding to X(j), FERR(j)
 *          is an estimated upper bound for the magnitude of the largest
 *          element in (X(j) - XTRUE) divided by the magnitude of the
 *          largest element in X(j).  The estimate is as reliable as
 *          the estimate for RCOND, and is almost always a slight
 *          overestimate of the true error.
 *
 *  BERR    (output) DOUBLE PRECISION array, dimension (NRHS)
 *          The componentwise relative backward error of each solution
 *          vector X(j) (i.e., the smallest relative change in
 *          any element of A or B that makes X(j) an exact solution).
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (3*N)
 *
 *  IWORK   (workspace) INTEGER array, dimension (N)
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRRFS(char uplo, char trans, char diag, int n, int nrhs, double* a, int lda, double* b, int ldb, double* x, int ldx, double* ferr, double* berr, double* work, int* iwork)
{
    int info;
    ::F_DTRRFS(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRSEN reorders the real Schur factorization of a real matrix
 *  A = Q*T*Q**T, so that a selected cluster of eigenvalues appears in
 *  the leading diagonal blocks of the upper quasi-triangular matrix T,
 *  and the leading columns of Q form an orthonormal basis of the
 *  corresponding right invariant subspace.
 *
 *  Optionally the routine computes the reciprocal condition numbers of
 *  the cluster of eigenvalues and/or the invariant subspace.
 *
 *  T must be in Schur canonical form (as returned by DHSEQR), that is,
 *  block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 *  2-by-2 diagonal block has its diagonal elemnts equal and its
 *  off-diagonal elements of opposite sign.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies whether condition numbers are required for the
 *          cluster of eigenvalues (S) or the invariant subspace (SEP):
 *          = 'N': none;
 *          = 'E': for eigenvalues only (S);
 *          = 'V': for invariant subspace only (SEP);
 *          = 'B': for both eigenvalues and invariant subspace (S and
 *                 SEP).
 *
 *  COMPQ   (input) CHARACTER*1
 *          = 'V': update the matrix Q of Schur vectors;
 *          = 'N': do not update Q.
 *
 *  SELECT  (input) LOGICAL array, dimension (N)
 *          SELECT specifies the eigenvalues in the selected cluster. To
 *          select a real eigenvalue w(j), SELECT(j) must be set to
 *          .TRUE.. To select a complex conjugate pair of eigenvalues
 *          w(j) and w(j+1), corresponding to a 2-by-2 diagonal block,
 *          either SELECT(j) or SELECT(j+1) or both must be set to
 *          .TRUE.; a complex conjugate pair of eigenvalues must be
 *          either both included in the cluster or both excluded.
 *
 *  N       (input) INTEGER
 *          The order of the matrix T. N >= 0.
 *
 *  T       (input/output) DOUBLE PRECISION array, dimension (LDT,N)
 *          On entry, the upper quasi-triangular matrix T, in Schur
 *          canonical form.
 *          On exit, T is overwritten by the reordered matrix T, again in
 *          Schur canonical form, with the selected eigenvalues in the
 *          leading diagonal blocks.
 *
 *  LDT     (input) INTEGER
 *          The leading dimension of the array T. LDT >= max(1,N).
 *
 *  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
 *          On entry, if COMPQ = 'V', the matrix Q of Schur vectors.
 *          On exit, if COMPQ = 'V', Q has been postmultiplied by the
 *          orthogonal transformation matrix which reorders T; the
 *          leading M columns of Q form an orthonormal basis for the
 *          specified invariant subspace.
 *          If COMPQ = 'N', Q is not referenced.
 *
 *  LDQ     (input) INTEGER
 *          The leading dimension of the array Q.
 *          LDQ >= 1; and if COMPQ = 'V', LDQ >= N.
 *
 *  WR      (output) DOUBLE PRECISION array, dimension (N)
 *  WI      (output) DOUBLE PRECISION array, dimension (N)
 *          The real and imaginary parts, respectively, of the reordered
 *          eigenvalues of T. The eigenvalues are stored in the same
 *          order as on the diagonal of T, with WR(i) = T(i,i) and, if
 *          T(i:i+1,i:i+1) is a 2-by-2 diagonal block, WI(i) > 0 and
 *          WI(i+1) = -WI(i). Note that if a complex eigenvalue is
 *          sufficiently ill-conditioned, then its value may differ
 *          significantly from its value before reordering.
 *
 *  M       (output) INTEGER
 *          The dimension of the specified invariant subspace.
 *          0 < = M <= N.
 *
 *  S       (output) DOUBLE PRECISION
 *          If JOB = 'E' or 'B', S is a lower bound on the reciprocal
 *          condition number for the selected cluster of eigenvalues.
 *          S cannot underestimate the true reciprocal condition number
 *          by more than a factor of sqrt(N). If M = 0 or N, S = 1.
 *          If JOB = 'N' or 'V', S is not referenced.
 *
 *  SEP     (output) DOUBLE PRECISION
 *          If JOB = 'V' or 'B', SEP is the estimated reciprocal
 *          condition number of the specified invariant subspace. If
 *          M = 0 or N, SEP = norm(T).
 *          If JOB = 'N' or 'E', SEP is not referenced.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.
 *          If JOB = 'N', LWORK >= max(1,N);
 *          if JOB = 'E', LWORK >= max(1,M*(N-M));
 *          if JOB = 'V' or 'B', LWORK >= max(1,2*M*(N-M)).
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  IWORK   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
 *          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
 *
 *  LIWORK  (input) INTEGER
 *          The dimension of the array IWORK.
 *          If JOB = 'N' or 'E', LIWORK >= 1;
 *          if JOB = 'V' or 'B', LIWORK >= max(1,M*(N-M)).
 *
 *          If LIWORK = -1, then a workspace query is assumed; the
 *          routine only calculates the optimal size of the IWORK array,
 *          returns this value as the first entry of the IWORK array, and
 *          no error message related to LIWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          = 1: reordering of T failed because some eigenvalues are too
 *               close to separate (the problem is very ill-conditioned);
 *               T may have been partially reordered, and WR and WI
 *               contain the eigenvalues in the same order as in T; S and
 *               SEP (if requested) are set to zero.
 *
 *  Further Details
 *  ===============
 *
 *  DTRSEN first collects the selected eigenvalues by computing an
 *  orthogonal transformation Z to move them to the top left corner of T.
 *  In other words, the selected eigenvalues are the eigenvalues of T11
 *  in:
 *
 *                Z'*T*Z = ( T11 T12 ) n1
 *                         (  0  T22 ) n2
 *                            n1  n2
 *
 *  where N = n1+n2 and Z' means the transpose of Z. The first n1 columns
 *  of Z span the specified invariant subspace of T.
 *
 *  If T has been obtained from the real Schur factorization of a matrix
 *  A = Q*T*Q', then the reordered real Schur factorization of A is given
 *  by A = (Q*Z)*(Z'*T*Z)*(Q*Z)', and the first n1 columns of Q*Z span
 *  the corresponding invariant subspace of A.
 *
 *  The reciprocal condition number of the average of the eigenvalues of
 *  T11 may be returned in S. S lies between 0 (very badly conditioned)
 *  and 1 (very well conditioned). It is computed as follows. First we
 *  compute R so that
 *
 *                         P = ( I  R ) n1
 *                             ( 0  0 ) n2
 *                               n1 n2
 *
 *  is the projector on the invariant subspace associated with T11.
 *  R is the solution of the Sylvester equation:
 *
 *                        T11*R - R*T22 = T12.
 *
 *  Let F-norm(M) denote the Frobenius-norm of M and 2-norm(M) denote
 *  the two-norm of M. Then S is computed as the lower bound
 *
 *                      (1 + F-norm(R)**2)**(-1/2)
 *
 *  on the reciprocal of 2-norm(P), the true reciprocal condition number.
 *  S cannot underestimate 1 / 2-norm(P) by more than a factor of
 *  sqrt(N).
 *
 *  An approximate error bound for the computed average of the
 *  eigenvalues of T11 is
 *
 *                         EPS * norm(T) / S
 *
 *  where EPS is the machine precision.
 *
 *  The reciprocal condition number of the right invariant subspace
 *  spanned by the first n1 columns of Z (or of Q*Z) is returned in SEP.
 *  SEP is defined as the separation of T11 and T22:
 *
 *                     sep( T11, T22 ) = sigma-min( C )
 *
 *  where sigma-min(C) is the smallest singular value of the
 *  n1*n2-by-n1*n2 matrix
 *
 *     C  = kprod( I(n2), T11 ) - kprod( transpose(T22), I(n1) )
 *
 *  I(m) is an m by m identity matrix, and kprod denotes the Kronecker
 *  product. We estimate sigma-min(C) by the reciprocal of an estimate of
 *  the 1-norm of inverse(C). The true reciprocal 1-norm of inverse(C)
 *  cannot differ from sigma-min(C) by more than a factor of sqrt(n1*n2).
 *
 *  When SEP is small, small changes in T can cause large changes in
 *  the invariant subspace. An approximate bound on the maximum angular
 *  error in the computed right invariant subspace is
 *
 *                      EPS * norm(T) / SEP
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRSEN(char job, char compq, int n, double* t, int ldt, double* q, int ldq, double* wr, double* wi, int* m, double* s, double* sep, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DTRSEN(&job, &compq, &n, t, &ldt, q, &ldq, wr, wi, m, s, sep, work, &lwork, iwork, &liwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRSNA estimates reciprocal condition numbers for specified
 *  eigenvalues and/or right eigenvectors of a real upper
 *  quasi-triangular matrix T (or of any matrix Q*T*Q**T with Q
 *  orthogonal).
 *
 *  T must be in Schur canonical form (as returned by DHSEQR), that is,
 *  block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 *  2-by-2 diagonal block has its diagonal elements equal and its
 *  off-diagonal elements of opposite sign.
 *
 *  Arguments
 *  =========
 *
 *  JOB     (input) CHARACTER*1
 *          Specifies whether condition numbers are required for
 *          eigenvalues (S) or eigenvectors (SEP):
 *          = 'E': for eigenvalues only (S);
 *          = 'V': for eigenvectors only (SEP);
 *          = 'B': for both eigenvalues and eigenvectors (S and SEP).
 *
 *  HOWMNY  (input) CHARACTER*1
 *          = 'A': compute condition numbers for all eigenpairs;
 *          = 'S': compute condition numbers for selected eigenpairs
 *                 specified by the array SELECT.
 *
 *  SELECT  (input) LOGICAL array, dimension (N)
 *          If HOWMNY = 'S', SELECT specifies the eigenpairs for which
 *          condition numbers are required. To select condition numbers
 *          for the eigenpair corresponding to a real eigenvalue w(j),
 *          SELECT(j) must be set to .TRUE.. To select condition numbers
 *          corresponding to a complex conjugate pair of eigenvalues w(j)
 *          and w(j+1), either SELECT(j) or SELECT(j+1) or both, must be
 *          set to .TRUE..
 *          If HOWMNY = 'A', SELECT is not referenced.
 *
 *  N       (input) INTEGER
 *          The order of the matrix T. N >= 0.
 *
 *  T       (input) DOUBLE PRECISION array, dimension (LDT,N)
 *          The upper quasi-triangular matrix T, in Schur canonical form.
 *
 *  LDT     (input) INTEGER
 *          The leading dimension of the array T. LDT >= max(1,N).
 *
 *  VL      (input) DOUBLE PRECISION array, dimension (LDVL,M)
 *          If JOB = 'E' or 'B', VL must contain left eigenvectors of T
 *          (or of any Q*T*Q**T with Q orthogonal), corresponding to the
 *          eigenpairs specified by HOWMNY and SELECT. The eigenvectors
 *          must be stored in consecutive columns of VL, as returned by
 *          DHSEIN or DTREVC.
 *          If JOB = 'V', VL is not referenced.
 *
 *  LDVL    (input) INTEGER
 *          The leading dimension of the array VL.
 *          LDVL >= 1; and if JOB = 'E' or 'B', LDVL >= N.
 *
 *  VR      (input) DOUBLE PRECISION array, dimension (LDVR,M)
 *          If JOB = 'E' or 'B', VR must contain right eigenvectors of T
 *          (or of any Q*T*Q**T with Q orthogonal), corresponding to the
 *          eigenpairs specified by HOWMNY and SELECT. The eigenvectors
 *          must be stored in consecutive columns of VR, as returned by
 *          DHSEIN or DTREVC.
 *          If JOB = 'V', VR is not referenced.
 *
 *  LDVR    (input) INTEGER
 *          The leading dimension of the array VR.
 *          LDVR >= 1; and if JOB = 'E' or 'B', LDVR >= N.
 *
 *  S       (output) DOUBLE PRECISION array, dimension (MM)
 *          If JOB = 'E' or 'B', the reciprocal condition numbers of the
 *          selected eigenvalues, stored in consecutive elements of the
 *          array. For a complex conjugate pair of eigenvalues two
 *          consecutive elements of S are set to the same value. Thus
 *          S(j), SEP(j), and the j-th columns of VL and VR all
 *          correspond to the same eigenpair (but not in general the
 *          j-th eigenpair, unless all eigenpairs are selected).
 *          If JOB = 'V', S is not referenced.
 *
 *  SEP     (output) DOUBLE PRECISION array, dimension (MM)
 *          If JOB = 'V' or 'B', the estimated reciprocal condition
 *          numbers of the selected eigenvectors, stored in consecutive
 *          elements of the array. For a complex eigenvector two
 *          consecutive elements of SEP are set to the same value. If
 *          the eigenvalues cannot be reordered to compute SEP(j), SEP(j)
 *          is set to 0; this can only occur when the true value would be
 *          very small anyway.
 *          If JOB = 'E', SEP is not referenced.
 *
 *  MM      (input) INTEGER
 *          The number of elements in the arrays S (if JOB = 'E' or 'B')
 *           and/or SEP (if JOB = 'V' or 'B'). MM >= M.
 *
 *  M       (output) INTEGER
 *          The number of elements of the arrays S and/or SEP actually
 *          used to store the estimated condition numbers.
 *          If HOWMNY = 'A', M is set to N.
 *
 *  WORK    (workspace) DOUBLE PRECISION array, dimension (LDWORK,N+6)
 *          If JOB = 'E', WORK is not referenced.
 *
 *  LDWORK  (input) INTEGER
 *          The leading dimension of the array WORK.
 *          LDWORK >= 1; and if JOB = 'V' or 'B', LDWORK >= N.
 *
 *  IWORK   (workspace) INTEGER array, dimension (2*(N-1))
 *          If JOB = 'E', IWORK is not referenced.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The reciprocal of the condition number of an eigenvalue lambda is
 *  defined as
 *
 *          S(lambda) = |v'*u| / (norm(u)*norm(v))
 *
 *  where u and v are the right and left eigenvectors of T corresponding
 *  to lambda; v' denotes the conjugate-transpose of v, and norm(u)
 *  denotes the Euclidean norm. These reciprocal condition numbers always
 *  lie between zero (very badly conditioned) and one (very well
 *  conditioned). If n = 1, S(lambda) is defined to be 1.
 *
 *  An approximate error bound for a computed eigenvalue W(i) is given by
 *
 *                      EPS * norm(T) / S(i)
 *
 *  where EPS is the machine precision.
 *
 *  The reciprocal of the condition number of the right eigenvector u
 *  corresponding to lambda is defined as follows. Suppose
 *
 *              T = ( lambda  c  )
 *                  (   0    T22 )
 *
 *  Then the reciprocal condition number is
 *
 *          SEP( lambda, T22 ) = sigma-min( T22 - lambda*I )
 *
 *  where sigma-min denotes the smallest singular value. We approximate
 *  the smallest singular value by the reciprocal of an estimate of the
 *  one-norm of the inverse of T22 - lambda*I. If n = 1, SEP(1) is
 *  defined to be abs(T(1,1)).
 *
 *  An approximate error bound for a computed right eigenvector VR(i)
 *  is given by
 *
 *                      EPS * norm(T) / SEP(i)
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRSNA(char job, char howmny, int n, double* t, int ldt, double* vl, int ldvl, double* vr, int ldvr, double* s, double* sep, int mm, int* m, double* work, int ldwork, int* iwork)
{
    int info;
    ::F_DTRSNA(&job, &howmny, &n, t, &ldt, vl, &ldvl, vr, &ldvr, s, sep, &mm, m, work, &ldwork, iwork, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRSYL solves the real Sylvester matrix equation:
 *
 *     op(A)*X + X*op(B) = scale*C or
 *     op(A)*X - X*op(B) = scale*C,
 *
 *  where op(A) = A or A**T, and  A and B are both upper quasi-
 *  triangular. A is M-by-M and B is N-by-N; the right hand side C and
 *  the solution X are M-by-N; and scale is an output scale factor, set
 *  <= 1 to avoid overflow in X.
 *
 *  A and B must be in Schur canonical form (as returned by DHSEQR), that
 *  is, block upper triangular with 1-by-1 and 2-by-2 diagonal blocks;
 *  each 2-by-2 diagonal block has its diagonal elements equal and its
 *  off-diagonal elements of opposite sign.
 *
 *  Arguments
 *  =========
 *
 *  TRANA   (input) CHARACTER*1
 *          Specifies the option op(A):
 *          = 'N': op(A) = A    (No transpose)
 *          = 'T': op(A) = A**T (Transpose)
 *          = 'C': op(A) = A**H (Conjugate transpose = Transpose)
 *
 *  TRANB   (input) CHARACTER*1
 *          Specifies the option op(B):
 *          = 'N': op(B) = B    (No transpose)
 *          = 'T': op(B) = B**T (Transpose)
 *          = 'C': op(B) = B**H (Conjugate transpose = Transpose)
 *
 *  ISGN    (input) INTEGER
 *          Specifies the sign in the equation:
 *          = +1: solve op(A)*X + X*op(B) = scale*C
 *          = -1: solve op(A)*X - X*op(B) = scale*C
 *
 *  M       (input) INTEGER
 *          The order of the matrix A, and the number of rows in the
 *          matrices X and C. M >= 0.
 *
 *  N       (input) INTEGER
 *          The order of the matrix B, and the number of columns in the
 *          matrices X and C. N >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,M)
 *          The upper quasi-triangular matrix A, in Schur canonical form.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A. LDA >= max(1,M).
 *
 *  B       (input) DOUBLE PRECISION array, dimension (LDB,N)
 *          The upper quasi-triangular matrix B, in Schur canonical form.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B. LDB >= max(1,N).
 *
 *  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)
 *          On entry, the M-by-N right hand side matrix C.
 *          On exit, C is overwritten by the solution matrix X.
 *
 *  LDC     (input) INTEGER
 *          The leading dimension of the array C. LDC >= max(1,M)
 *
 *  SCALE   (output) DOUBLE PRECISION
 *          The scale factor, scale, set <= 1 to avoid overflow in X.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          = 1: A and B have common or very close eigenvalues; perturbed
 *               values were used to solve the equation (but the matrices
 *               A and B are unchanged).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRSYL(char trana, char tranb, int isgn, int m, int n, double* a, int lda, double* b, int ldb, double* c, int ldc, double* scale)
{
    int info;
    ::F_DTRSYL(&trana, &tranb, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, scale, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRTRI computes the inverse of a real upper or lower triangular
 *  matrix A.
 *
 *  This is the Level 3 BLAS version of the algorithm.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the triangular matrix A.  If UPLO = 'U', the
 *          leading N-by-N upper triangular part of the array A contains
 *          the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading N-by-N lower triangular part of the array A contains
 *          the lower triangular matrix, and the strictly upper
 *          triangular part of A is not referenced.  If DIAG = 'U', the
 *          diagonal elements of A are also not referenced and are
 *          assumed to be 1.
 *          On exit, the (triangular) inverse of the original matrix, in
 *          the same storage format.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0: successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, A(i,i) is exactly zero.  The triangular
 *               matrix is singular and its inverse can not be computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRTRI(char uplo, char diag, int n, double* a, int lda)
{
    int info;
    ::F_DTRTRI(&uplo, &diag, &n, a, &lda, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTRTRS solves a triangular system of the form
 *
 *     A * X = B  or  A**T * X = B,
 *
 *  where A is a triangular matrix of order N, and B is an N-by-NRHS
 *  matrix.  A check is made to verify that A is nonsingular.
 *
 *  Arguments
 *  =========
 *
 *  UPLO    (input) CHARACTER*1
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 *  TRANS   (input) CHARACTER*1
 *          Specifies the form of the system of equations:
 *          = 'N':  A * X = B  (No transpose)
 *          = 'T':  A**T * X = B  (Transpose)
 *          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
 *
 *  DIAG    (input) CHARACTER*1
 *          = 'N':  A is non-unit triangular;
 *          = 'U':  A is unit triangular.
 *
 *  N       (input) INTEGER
 *          The order of the matrix A.  N >= 0.
 *
 *  NRHS    (input) INTEGER
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  NRHS >= 0.
 *
 *  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
 *          The triangular matrix A.  If UPLO = 'U', the leading N-by-N
 *          upper triangular part of the array A contains the upper
 *          triangular matrix, and the strictly lower triangular part of
 *          A is not referenced.  If UPLO = 'L', the leading N-by-N lower
 *          triangular part of the array A contains the lower triangular
 *          matrix, and the strictly upper triangular part of A is not
 *          referenced.  If DIAG = 'U', the diagonal elements of A are
 *          also not referenced and are assumed to be 1.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,N).
 *
 *  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
 *          On entry, the right hand side matrix B.
 *          On exit, if INFO = 0, the solution matrix X.
 *
 *  LDB     (input) INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,N).
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0: if INFO = -i, the i-th argument had an illegal value
 *          > 0: if INFO = i, the i-th diagonal element of A is zero,
 *               indicating that the matrix is singular and the solutions
 *               X have not been computed.
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTRTRS(char uplo, char trans, char diag, int n, int nrhs, double* a, int lda, double* b, int ldb)
{
    int info;
    ::F_DTRTRS(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  This routine is deprecated and has been replaced by routine DTZRZF.
 *
 *  DTZRQF reduces the M-by-N ( M<=N ) real upper trapezoidal matrix A
 *  to upper triangular form by means of orthogonal transformations.
 *
 *  The upper trapezoidal matrix A is factored as
 *
 *     A = ( R  0 ) * Z,
 *
 *  where Z is an N-by-N orthogonal matrix and R is an M-by-M upper
 *  triangular matrix.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= M.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the leading M-by-N upper trapezoidal part of the
 *          array A must contain the matrix to be factorized.
 *          On exit, the leading M-by-M upper triangular part of A
 *          contains the upper triangular matrix R, and elements M+1 to
 *          N of the first M rows of A, with the array TAU, represent the
 *          orthogonal matrix Z as a product of M elementary reflectors.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (M)
 *          The scalar factors of the elementary reflectors.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  The factorization is obtained by Householder's method.  The kth
 *  transformation matrix, Z( k ), which is used to introduce zeros into
 *  the ( m - k + 1 )th row of A, is given in the form
 *
 *     Z( k ) = ( I     0   ),
 *              ( 0  T( k ) )
 *
 *  where
 *
 *     T( k ) = I - tau*u( k )*u( k )',   u( k ) = (   1    ),
 *                                                 (   0    )
 *                                                 ( z( k ) )
 *
 *  tau is a scalar and z( k ) is an ( n - m ) element vector.
 *  tau and z( k ) are chosen to annihilate the elements of the kth row
 *  of X.
 *
 *  The scalar tau is returned in the kth element of TAU and the vector
 *  u( k ) in the kth row of A, such that the elements of z( k ) are
 *  in  a( k, m + 1 ), ..., a( k, n ). The elements of R are returned in
 *  the upper triangular part of A.
 *
 *  Z is given by
 *
 *     Z =  Z( 1 ) * Z( 2 ) * ... * Z( m ).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTZRQF(int m, int n, double* a, int lda, double* tau)
{
    int info;
    ::F_DTZRQF(&m, &n, a, &lda, tau, &info);
    return info;
}

/**
 *  Purpose
 *  =======
 *
 *  DTZRZF reduces the M-by-N ( M<=N ) real upper trapezoidal matrix A
 *  to upper triangular form by means of orthogonal transformations.
 *
 *  The upper trapezoidal matrix A is factored as
 *
 *     A = ( R  0 ) * Z,
 *
 *  where Z is an N-by-N orthogonal matrix and R is an M-by-M upper
 *  triangular matrix.
 *
 *  Arguments
 *  =========
 *
 *  M       (input) INTEGER
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  N       (input) INTEGER
 *          The number of columns of the matrix A.  N >= M.
 *
 *  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
 *          On entry, the leading M-by-N upper trapezoidal part of the
 *          array A must contain the matrix to be factorized.
 *          On exit, the leading M-by-M upper triangular part of A
 *          contains the upper triangular matrix R, and elements M+1 to
 *          N of the first M rows of A, with the array TAU, represent the
 *          orthogonal matrix Z as a product of M elementary reflectors.
 *
 *  LDA     (input) INTEGER
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  TAU     (output) DOUBLE PRECISION array, dimension (M)
 *          The scalar factors of the elementary reflectors.
 *
 *  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
 *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
 *
 *  LWORK   (input) INTEGER
 *          The dimension of the array WORK.  LWORK >= max(1,M).
 *          For optimum performance LWORK >= M*NB, where NB is
 *          the optimal blocksize.
 *
 *          If LWORK = -1, then a workspace query is assumed; the routine
 *          only calculates the optimal size of the WORK array, returns
 *          this value as the first entry of the WORK array, and no error
 *          message related to LWORK is issued by XERBLA.
 *
 *  C++ Return value: INFO    (output) INTEGER
 *          = 0:  successful exit
 *          < 0:  if INFO = -i, the i-th argument had an illegal value
 *
 *  Further Details
 *  ===============
 *
 *  Based on contributions by
 *    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA
 *
 *  The factorization is obtained by Householder's method.  The kth
 *  transformation matrix, Z( k ), which is used to introduce zeros into
 *  the ( m - k + 1 )th row of A, is given in the form
 *
 *     Z( k ) = ( I     0   ),
 *              ( 0  T( k ) )
 *
 *  where
 *
 *     T( k ) = I - tau*u( k )*u( k )',   u( k ) = (   1    ),
 *                                                 (   0    )
 *                                                 ( z( k ) )
 *
 *  tau is a scalar and z( k ) is an ( n - m ) element vector.
 *  tau and z( k ) are chosen to annihilate the elements of the kth row
 *  of X.
 *
 *  The scalar tau is returned in the kth element of TAU and the vector
 *  u( k ) in the kth row of A, such that the elements of z( k ) are
 *  in  a( k, m + 1 ), ..., a( k, n ). The elements of R are returned in
 *  the upper triangular part of A.
 *
 *  Z is given by
 *
 *     Z =  Z( 1 ) * Z( 2 ) * ... * Z( m ).
 *
 *  =====================================================================
 *
 *     .. Parameters ..
 **/
int C_DTZRZF(int m, int n, double* a, int lda, double* tau, double* work, int lwork)
{
    int info;
    ::F_DTZRZF(&m, &n, a, &lda, tau, work, &lwork, &info);
    return info;
}

int C_DGESV(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb)
{
    int info;
    ::F_DGESV(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

int C_DSYEV(char jobz, char uplo, int n, double* a, int lda, double* w, double* work, int lwork)
{
    int info;
    ::F_DSYEV(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

int C_DSYEVD(char jobz, char uplo, int n, double* a, int lda, double* w, double* work, int lwork, int* iwork, int liwork)
{
    int info;
    ::F_DSYEVD(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
    return info;
}

int C_DGESVD(char jobu, char jobv, int m, int n, double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt, double* work, int lwork)
{
    int info;
    ::F_DGESVD(&jobu, &jobv, &m, &n, A, &lda, s, U, &ldu, VT, &ldvt, work, &lwork, &info);
    return info;
}

int C_DGESDD(char jobz, int m, int n, double* A, int lda, double* s, double* U, int ldu, double* VT, int ldvt, double* work, int lwork, int* iwork)
{
    int info;
    ::F_DGESDD(&jobz, &m, &n, A, &lda, s, U, &ldu, VT, &ldvt, work, &lwork, iwork, &info);
    return info;
}

} // namespace lightspeed
