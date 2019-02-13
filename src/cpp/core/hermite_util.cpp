#include <lightspeed/hermite.hpp>
#include <cmath>

namespace lightspeed {

std::vector<int> HermiteUtil::angl(int H)
{
    std::vector<int> val;
    for (int L = 0; L <= H; L++) {
        for (int i = 0; i <= L; i++) {
            int l = L - i;
        for (int j = 0; j <= i; j++) {
            int m = i - j;
            int n = j;
            int ang = (l << 20) + (m << 10) + (n << 0);
            val.push_back(ang);
        }}
    }
    return val;
}    

std::vector<double> HermiteUtil::dmax_bra(
    const HermiteL& herm)
{
    size_t npair = herm.npair();
    std::vector<double> val(npair);

    const PairListL* pairlist = herm.pairs();
    const std::vector<Pair>& pairs = pairlist->pairs(); 
    
    for (size_t ind = 0; ind < npair; ind++) {
        const Pair& pair = pairs[ind];
        const Primitive& pair1 = pair.prim1();
        const Primitive& pair2 = pair.prim2();
        double K = fabs(pair1.c() * pair2.c() * exp(-
            pair1.e() * pair2.e() / (pair1.e() + pair2.e()) * (
            pow(pair1.x() - pair2.x(),2) +
            pow(pair1.y() - pair2.y(),2) +
            pow(pair1.z() - pair2.z(),2))));
        val[ind] = K;
    }
    return val;
}

std::vector<double> HermiteUtil::dmax_ket(
    const HermiteL& herm)
{
    const double* dp = herm.data().data();
    size_t npair = herm.npair();
    int H = herm.H();
    std::vector<double> val(npair);
    for (size_t ind = 0; ind < npair; ind++) {
        double dval = 0.0;
        for (int ind2 = 0; ind2 < H; ind2++) {
            dval = std::max(dval, fabs(*dp++));
        }
        val[ind] = dval;
    }
    return val;
}

} // namespace lightspeed
