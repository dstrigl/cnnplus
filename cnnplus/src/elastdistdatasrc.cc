/**************************************************************************//**
 *
 * \file   elastdistdatasrc.cc
 * \author Daniel Strigl, Klaus Kofler
 * \date   Apr 25 2009
 *
 * $Id: elastdistdatasrc.cc 1827 2009-07-20 18:17:58Z dast $
 *
 * \brief  Implementation of cnnplus::ElastDistDataSrc.
 *
 *****************************************************************************/

#include "elastdistdatasrc.hh"
#include "error.hh"
#include "matvecli.hh"
#include "mathli.hh"
#include <algorithm>
#include <sstream>

CNNPLUS_NS_BEGIN

static void
createGaussKernel(float * k, int const h, int const w, float const sigma)
{
    CNNPLUS_ASSERT(k && h > 0 && h % 2 != 0 && w > 0 && w % 2 != 0);
    CNNPLUS_ASSERT(sigma > 0);

    float sum = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float const n1 = static_cast<float>(y - h/2);
            float const n2 = static_cast<float>(x - w/2);
            sum += (k[y * w + x] = mathli::exp(-(n1*n1 + n2*n2)/(2*sigma*sigma)));
        }
    }

    matvecli::divvc<float>(k, sum, h * w);
}

static void
filter(float const * in, float * out, int const rows, int const cols,
       float const * k, int const h, int const w, float const alpha)
{
    CNNPLUS_ASSERT(in && out && in != out && rows > 0 && cols > 0);
    CNNPLUS_ASSERT(k && h > 0 && h % 2 != 0 && w > 0 && w % 2 != 0);

    int const h2 = h / 2;
    int const w2 = w / 2;

    for (int r = 0; r < rows; ++r) {

        int const y1 = mathli::max(r - h2, 0);
        int const y2 = mathli::min(r + h2 + 1, rows);
        int const j1 = mathli::max(h2 - r, 0);

        for (int c = 0; c < cols; ++c) {

            int const x1 = mathli::max(c - w2, 0);
            int const x2 = mathli::min(c + w2 + 1, cols);
            int const i1 = mathli::max(w2 - c, 0);

            float sum = 0;

            for (int y = y1, j = j1; y < y2; ++y, ++j)
                for (int x = x1, i = i1; x < x2; ++x, ++i)
                    sum += in[y * cols + x] * k[j * w + i];

            out[r * cols + c] = sum * alpha;
        }
    }
}

//! Helper macro for pixel value at position (x,y)
#define pixAt(x,y) ((x < 0 || x >= cols) ? background : \
                    (y < 0 || y >= rows) ? background : \
                    *(in + y * cols + x))

template<typename T> void
distort(T const * in, T * out, float const * dx, float const * dy,
        int const rows, int const cols, T const background)
{
    CNNPLUS_ASSERT(in && out && in != out && dx && dy);
    CNNPLUS_ASSERT(rows > 0 && cols > 0);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {

            // This bilinear interpolation code is based on the implementation
            // in "The CImg Library - C++ Template Image Processing Toolkit",
            // which can be found at http://cimg.sourceforge.net/.

            float const
                fx = c + *(dx++),
                fy = r + *(dy++);

            int const
                x = static_cast<int>(fx) - (fx >= 0 ? 0 : 1), nx = x + 1,
                y = static_cast<int>(fy) - (fy >= 0 ? 0 : 1), ny = y + 1;

            float const
                dx = fx - x,
                dy = fy - y;

            T const
                Icc = pixAt(x,  y), Inc = pixAt(nx,  y),
                Icn = pixAt(x, ny), Inn = pixAt(nx, ny);

            *(out++) = Icc + dx*(Inc-Icc + dy*(Icc+Inn-Icn-Inc)) + dy*(Icn-Icc);
        }
    }
}

template<typename T>
ElastDistDataSrc<T>::ElastDistDataSrc(DataSource<T> & ds, int const addSamples,
    float const alpha, float const sigma, Size const & sizeKernel)
    : DataSource<T>(ds.dataRangeMin(), ds.dataRangeMax(), ds.size() * (addSamples + 1), 0),
    ds_(ds), addSamples_(addSamples), alpha_(alpha), sigma_(sigma),
    sizeKernel_(sizeKernel), dx_(NULL), dy_(NULL), tmp_(NULL), idx_()
{
    if (addSamples < 0)
        throw ParameterError("addSamples", "must be greater or equal to zero.");
    else if (alpha <= 0)
        throw ParameterError("alpha", "must be greater zero.");
    else if (sigma <= 0)
        throw ParameterError("sigma", "must be greater zero.");
    else if (sizeKernel.height() == 0 || sizeKernel.height() % 2 == 0)
        throw ParameterError("sizeKernel", "height must be greater zero and odd.");
    else if (sizeKernel.width() == 0 || sizeKernel.width() % 2 == 0)
        throw ParameterError("sizeKernel", "width must be greater zero and odd.");

    if (addSamples_ > 0) {
        dx_  = matvecli::allocv<float>(ds_.sizeOut() * addSamples_);
        dy_  = matvecli::allocv<float>(ds_.sizeOut() * addSamples_);
        tmp_ = matvecli::allocv<T>(ds_.sizeOut());
        createDisplacementFields();
    }

    reset();
}

template<typename T>
ElastDistDataSrc<T>::~ElastDistDataSrc()
{
    if (dx_ ) matvecli::free<float>(dx_);
    if (dy_ ) matvecli::free<float>(dy_);
    if (tmp_) matvecli::free<T>(tmp_);
}

template<typename T> void
ElastDistDataSrc<T>::createDisplacementFields()
{
    float * kernel = matvecli::allocv<float>(sizeKernel_.area());
    createGaussKernel(kernel, sizeKernel_.height(), sizeKernel_.width(), sigma_);

    float * dx = matvecli::allocv<float>(ds_.sizeOut());
    float * dy = matvecli::allocv<float>(ds_.sizeOut());

    matvecli::randreseed();

    for (int i = 0; i < addSamples_; ++i) {

        matvecli::randv<float>(dx, ds_.sizeOut(), 1);
        matvecli::randv<float>(dy, ds_.sizeOut(), 1);

        for (size_t j = 0; j < ds_.sizeOut(); ++j) {
            float const n = mathli::sqrt(dx[j] * dx[j] + dy[j] * dy[j]);
            dx[j] /= n;
            dy[j] /= n;
        }

        filter(dx, dx_ + i * ds_.sizeOut(),
               ds_.sizePattern().height(), ds_.sizePattern().width(),
               kernel, sizeKernel_.height(), sizeKernel_.width(), alpha_);
        filter(dy, dy_ + i * ds_.sizeOut(),
               ds_.sizePattern().height(), ds_.sizePattern().width(),
               kernel, sizeKernel_.height(), sizeKernel_.width(), alpha_);
    }

    matvecli::free<float>(kernel);
    matvecli::free<float>(dx);
    matvecli::free<float>(dy);
}

template<typename T> int
ElastDistDataSrc<T>::fprop(T * out)
{
    ds_.seek(idx_[this->curPos_] % ds_.size());
    int field = idx_[this->curPos_] / ds_.size();

    if (field-- > 0) {
        CNNPLUS_ASSERT(tmp_ && dx_ && dy_);
        CNNPLUS_ASSERT(field >= 0 && field < addSamples_);
        int const label = ds_.fprop(tmp_);
        distort<T>(tmp_, out,
            dx_ + field * ds_.sizeOut(), dy_ + field * ds_.sizeOut(),
            ds_.sizePattern().height(), ds_.sizePattern().width(),
            ds_.dataRangeMin());
        return label;
    }

    return ds_.fprop(out);
}

template<typename T> int
ElastDistDataSrc<T>::seek(int pos)
{
    if (pos < 0 || pos >= this->size_) {
        std::stringstream ss;
        ss << "must be between 0 and " << (this->size_ - 1) << ".";
        throw ParameterError("pos", ss.str());
    }

    int const oldPos = this->curPos_;
    this->curPos_ = pos;
    return oldPos;
}

template<typename T> void
ElastDistDataSrc<T>::shuffle()
{
    std::random_shuffle(idx_.begin(), idx_.end());
}

template<typename T> void
ElastDistDataSrc<T>::shift(int dist)
{
    if (this->size_ > 0) {
        while (dist < 0) dist += this->size_;
        for (int i = 0; i < this->size_; ++i)
            idx_[i] = (idx_[i] + dist) % this->size_;
    }
}

template<typename T> void
ElastDistDataSrc<T>::reset()
{
    if (this->size_ > 0) {
        idx_.resize(this->size_);
        for (int i = 0; i < this->size_; ++i)
            idx_[i] = i;
    }
}

template<typename T> size_t
ElastDistDataSrc<T>::sizeOut() const
{
    return ds_.sizeOut();
}

template<typename T> Size
ElastDistDataSrc<T>::sizePattern() const
{
    return ds_.sizePattern();
}

template<typename T> int
ElastDistDataSrc<T>::idx(int pos) const
{
    pos = (pos < 0) ? this->curPos_ : pos;

    if (pos < 0 || pos >= this->size_) {
        std::stringstream ss;
        ss << "must be between 0 and " << (this->size_ - 1) << ".";
        throw ParameterError("pos", ss.str());
    }

    return ds_.idx(idx_[pos] % ds_.size());
}

template<typename T> std::string
ElastDistDataSrc<T>::toString() const
{
    std::stringstream ss;
    ss << "ElastDistDataSrc["
        << ds_.toString()
        << "; addSamples=" << addSamples_
        << ", sigma=" << sigma_
        << ", alpha=" << alpha_
        << ", size=" << this->size_
        << "]";
    return ss.str();
}

/*! \addtogroup eti_grp Explicit Template Instantiation
 @{
 */
template class ElastDistDataSrc<float>;
template class ElastDistDataSrc<double>;
/*! @} */

CNNPLUS_NS_END
