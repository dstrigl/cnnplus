/**************************************************************************//**
 *
 * \file   elastdistdatasrc.hh
 * \author Daniel Strigl, Klaus Kofler
 * \date   Apr 25 2009
 *
 * $Id: elastdistdatasrc.hh 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Header for cnnplus::ElastDistDataSrc.
 *
 *****************************************************************************/

#ifndef CNNPLUS_ELASTDISTDATASRC_HH
#define CNNPLUS_ELASTDISTDATASRC_HH

#include "common.hh"
#include "datasource.hh"
#include <vector>

CNNPLUS_NS_BEGIN

//! \todo doc
/*! \see P. Y. Simard, D. Steinkraus, & J. Platt, "Best Practice for
         Convolutional Neural Networks Applied to Visual Document Analysis",
         International Conference on Document Analysis and Recognition (ICDAR),
         IEEE Computer Society, Los Alamitos, 2003, pp. 958-962.
 */
template<typename T>
class ElastDistDataSrc : public DataSource<T>
{
    void createDisplacementFields();

public:
    //! Ctr
    ElastDistDataSrc(DataSource<T> & ds, int addSamples = 9,
        float alpha = 34, float sigma = 4, Size const & sizeKernel = Size(21, 21));
    //! Dtr
    virtual ~ElastDistDataSrc();

    virtual int fprop(T * out);
    virtual int seek(int pos);
    virtual void shuffle();
    virtual void shift(int dist);
    virtual void reset();
    virtual size_t sizeOut() const;
    virtual Size sizePattern() const;
    virtual int idx(int pos = -1) const;
    virtual std::string toString() const;

    int addSamples() const { return addSamples_; }
    Size sizeKernel() const { return sizeKernel_; }
    float sigma() const { return sigma_; }
    float alpha() const { return alpha_; }

    //! Refresh the displacement fields
    void refresh() { createDisplacementFields(); }

private:
    DataSource<T> & ds_;
    int const addSamples_;
    float const alpha_;
    float const sigma_;
    Size const sizeKernel_;
    float * dx_;
    float * dy_;
    T * tmp_;
    std::vector<int> idx_;
};

CNNPLUS_NS_END

#endif // CNNPLUS_ELASTDISTDATASRC_HH
