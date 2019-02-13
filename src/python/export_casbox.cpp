#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <memory>
#include <lightspeed/casbox.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/resource_list.hpp>

using namespace lightspeed;
using namespace boost::python;

void export_casbox()
{
    class_<CASBox, std::shared_ptr<CASBox> >("CASBox", init<
        int,
        int,
        int,
        const std::shared_ptr<Tensor>&,
        const std::shared_ptr<Tensor>&
        >())
        .add_property("M", &CASBox::M)
        .add_property("Na", &CASBox::Na)
        .add_property("Nb", &CASBox::Nb)
        .add_property("N", &CASBox::N)
        .add_property("Da", &CASBox::Da)
        .add_property("Db", &CASBox::Db)
        .add_property("D", &CASBox::D)
        .add_property("H", &CASBox::H)
        .add_property("I", &CASBox::I)
        .def("__str__", &CASBox::string)
        .add_property("stringsA", &CASBox::stringsA)
        .add_property("stringsB", &CASBox::stringsB)
        .add_property("min_seniority", &CASBox::min_seniority)
        .add_property("max_seniority", &CASBox::max_seniority)
        .add_property("seniority", &CASBox::seniority)
        .def("seniority_block", &CASBox::seniority_block)
        .def("CSF_basis", &CASBox::CSF_basis)
        .add_property("E0_evangelisti", &CASBox::E0_evangelisti)
        .add_property("F_evangelisti", &CASBox::F_evangelisti)
        .def("H_evangelisti", &CASBox::H_evangelisti)
        .def("apply_evangelisti", &CASBox::apply_evangelisti)
        .def("guess_evangelisti", &CASBox::guess_evangelisti)
        .def("sigma", &CASBox::sigma) 
        .def("opdm", &CASBox::opdm) 
        .def("tpdm", &CASBox::tpdm) 
        .def("sigma_det", &CASBox::sigma_det) 
        .def("opdm_det", &CASBox::opdm_det) 
        .def("tpdm_det", &CASBox::tpdm_det)
        .def("sigma_det_gpu", &CASBox::sigma_det_gpu)
        .def("sigma_det_gpu", &CASBox::opdm_det_gpu)
        .def("sigma_det_gpu", &CASBox::tpdm_det_gpu)
        .def("orbital_transformation_det", &CASBox::orbital_transformation_det)
        .def("amplitude_string", &CASBox::amplitude_string)
        .def("dyson_orbital_a", &CASBox::dyson_orbital_a)
        .staticmethod("dyson_orbital_a")
        .def("dyson_orbital_b", &CASBox::dyson_orbital_b)
        .staticmethod("dyson_orbital_b")
        .def("metric_det", &CASBox::metric_det)
        ;

    class_<SeniorityBlock, std::shared_ptr<SeniorityBlock> >("SeniorityBlock", init<
        int,
        int,
        int,
        int
        >())
        .add_property("M", &SeniorityBlock::M)
        .add_property("Na", &SeniorityBlock::Na)
        .add_property("Nb", &SeniorityBlock::Nb)
        .add_property("Z", &SeniorityBlock::Z)
        .add_property("H", &SeniorityBlock::H)
        .add_property("D", &SeniorityBlock::D)
        .add_property("U", &SeniorityBlock::U)
        .add_property("A", &SeniorityBlock::A)
        .add_property("B", &SeniorityBlock::B)
        .add_property("paired_strings", &SeniorityBlock::paired_strings)
        .add_property("unpaired_strings", &SeniorityBlock::unpaired_strings)
        .add_property("interleave_strings", &SeniorityBlock::interleave_strings)
        .add_property("npaired", &SeniorityBlock::npaired)
        .add_property("nunpaired", &SeniorityBlock::nunpaired)
        .add_property("ninterleave", &SeniorityBlock::ninterleave)
        .def("compute_S2", &SeniorityBlock::compute_S2)
        ;

    class_<CSFBasis, std::shared_ptr<CSFBasis> >("CSFBasis", no_init)
        .add_property("S", &CSFBasis::S)
        .def("__str__", &CSFBasis::string)
        .add_property("total_nCSF", &CSFBasis::total_nCSF)
        .add_property("total_ndet", &CSFBasis::total_ndet)
        .add_property("seniority", &CSFBasis::seniority)
        .add_property("det_to_CSF", &CSFBasis::det_to_CSF)
        .add_property("nCSF", &CSFBasis::nCSF)
        .add_property("nunpaired", &CSFBasis::nunpaired)
        .add_property("npaired", &CSFBasis::npaired)
        .add_property("ninterleave", &CSFBasis::ninterleave)
        .add_property("nblock", &CSFBasis::nblock)
        .add_property("offsets_CSF", &CSFBasis::offsets_CSF)
        .add_property("sizes_CSF", &CSFBasis::sizes_CSF)
        .add_property("offsets_det", &CSFBasis::offsets_det)
        .add_property("sizes_det", &CSFBasis::sizes_det)
        .add_property("unpaired_strings", &CSFBasis::unpaired_strings)
        .add_property("paired_strings", &CSFBasis::paired_strings)
        .add_property("interleave_strings", &CSFBasis::interleave_strings)
        .def("transform_det_to_CSF", &CSFBasis::transform_det_to_CSF)
        .def("transform_CSF_to_det", &CSFBasis::transform_CSF_to_det)
        .def("transform_det_to_det2", &CSFBasis::transform_det_to_det2)
        .def("transform_det2_to_det", &CSFBasis::transform_det2_to_det)
        .def("transform_det2_to_CSF", &CSFBasis::transform_det2_to_CSF)
        .def("transform_CSF_to_det2", &CSFBasis::transform_CSF_to_det2)
        ;

    class_<ExplicitCASBox, std::shared_ptr<ExplicitCASBox> >("ExplicitCASBox", init<
        const std::shared_ptr<CASBox>&
        >())
        .add_property("casbox", &ExplicitCASBox::casbox)
        .def("evecs", &ExplicitCASBox::evecs)
        .def("evals", &ExplicitCASBox::evals)
        .def("evec", &ExplicitCASBox::evec)
        .def("eval", &ExplicitCASBox::eval)
        ;
}
