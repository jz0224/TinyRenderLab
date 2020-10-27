#pragma once

#include "Basic/Util.h"
#include <iterator>
#include <assert.h>

template <typename T>
inline bool isNaN(const T x) {
    return std::isnan(x);
}
template <>
inline bool isNaN(const int x) {
    return false;
}

// Vector Declarations
template <typename T>
class Vector2 {
  public:
    // Vector2 Public Methods
    Vector2() { x = y = 0; }
    Vector2(T xx) : x(xx), y(xx) { assert(!HasNaNs()); }
    Vector2(T xx, T yy) : x(xx), y(yy) { assert(!HasNaNs()); }
    bool HasNaNs() const { return isNaN(x) || isNaN(y); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    Vector2(const Vector2<T> &v) {
        assert(!v.HasNaNs());
        x = v.x;
        y = v.y;
    }
    Vector2<T> &operator=(const Vector2<T> &v) {
        assert(!v.HasNaNs());
        x = v.x;
        y = v.y;
        return *this;
    }
#endif  // !NDEBUG

    Vector2<T> operator+(const Vector2<T> &v) const {
        assert(!v.HasNaNs());
        return Vector2(x + v.x, y + v.y);
    }

    Vector2<T> &operator+=(const Vector2<T> &v) {
        assert(!v.HasNaNs());
        x += v.x;
        y += v.y;
        return *this;
    }
    Vector2<T> operator-(const Vector2<T> &v) const {
        assert(!v.HasNaNs());
        return Vector2(x - v.x, y - v.y);
    }

    Vector2<T> &operator-=(const Vector2<T> &v) {
        assert(!v.HasNaNs());
        x -= v.x;
        y -= v.y;
        return *this;
    }
    bool operator==(const Vector2<T> &v) const { return x == v.x && y == v.y; }
    bool operator!=(const Vector2<T> &v) const { return x != v.x || y != v.y; }
    template <typename U>
    Vector2<T> operator*(U f) const {
        return Vector2<T>(f * x, f * y);
    }

    template <typename U>
    Vector2<T> &operator*=(U f) {
        assert(!isNaN(f));
        x *= f;
        y *= f;
        return *this;
    }
    template <typename U>
    Vector2<T> operator/(U f) const {
        assert(f != 0);
        float inv = (float)1 / f;
        return Vector2<T>(x * inv, y * inv);
    }

    template <typename U>
    Vector2<T> &operator/=(U f) {
        assert(f != 0);
        float inv = (float)1 / f;
        x *= inv;
        y *= inv;
        return *this;
    }
    Vector2<T> operator-() const { return Vector2<T>(-x, -y); }
    T operator[](int i) const {
        assert(i >= 0 && i <= 1);
        if (i == 0) return x;
        return y;
    }

    T &operator[](int i) {
        assert(i >= 0 && i <= 1);
        if (i == 0) return x;
        return y;
    }
    float LengthSquared() const { return x * x + y * y; }
    float Length() const { return std::sqrt(LengthSquared()); }

    // Vector2 Public Data
    T x, y;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
    os << "[ " << v.x << ", " << v.y << " ]";
    return os;
}

//template <>
//inline std::ostream &operator<<(std::ostream &os, const Vector2<float> &v) {
//    os << StringPrintf("[ %f, %f ]", v.x, v.y);
//    return os;
//}

template <typename T>
class Vector3 {
  public:
    // Vector3 Public Methods
    T operator[](int i) const {
        assert(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    T &operator[](int i) {
        assert(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    Vector3() { x = y = z = 0; }
	Vector3(T x) : x(x), y(x), z(x) { assert(!HasNaNs()); }
    Vector3(T x, T y, T z) : x(x), y(y), z(z) { assert(!HasNaNs()); }
    bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    Vector3(const Vector3<T> &v) {
        assert(!v.HasNaNs());
        x = v.x;
        y = v.y;
        z = v.z;
    }

    Vector3<T> &operator=(const Vector3<T> &v) {
        assert(!v.HasNaNs());
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
#endif  // !NDEBUG
    Vector3<T> operator+(const Vector3<T> &v) const {
        assert(!v.HasNaNs());
        return Vector3(x + v.x, y + v.y, z + v.z);
    }
    Vector3<T> &operator+=(const Vector3<T> &v) {
        assert(!v.HasNaNs());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vector3<T> operator-(const Vector3<T> &v) const {
        assert(!v.HasNaNs());
        return Vector3(x - v.x, y - v.y, z - v.z);
    }
    Vector3<T> &operator-=(const Vector3<T> &v) {
        assert(!v.HasNaNs());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    bool operator==(const Vector3<T> &v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator!=(const Vector3<T> &v) const {
        return x != v.x || y != v.y || z != v.z;
    }
    template <typename U>
    Vector3<T> operator*(U s) const {
        return Vector3<T>(s * x, s * y, s * z);
    }

	template <typename U>
    Vector3<T> operator*(Vector3<U> s) const {
        return Vector3<T>(s.x * x, s.y * y, s.z * z);
    }

    template <typename U>
    Vector3<T> &operator*=(U s) {
        assert(!isNaN(s));
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    template <typename U>
    Vector3<T> operator/(U f) const {
		assert(f != 0);
        float inv = (float)1 / f;
        return Vector3<T>(x * inv, y * inv, z * inv);
    }

    template <typename U>
    Vector3<T> &operator/=(U f) {
        assert(f != 0);
        float inv = (float)1 / f;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }
    Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }
    float LengthSquared() const { return x * x + y * y + z * z; }
    float Length() const { return std::sqrt(LengthSquared()); }

    // Vector3 Public Data
    T x, y, z;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
    os << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
    return os;
}

//template <>
//inline std::ostream &operator<<(std::ostream &os, const Vector3<float> &v) {
//    os << StringPrintf("[ %f, %f, %f ]", v.x, v.y, v.z);
//    return os;
//}

typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;

inline Vector3f Lerp(const float &a, const float &b, const float &t)
{
	return a * (1 - t) + b * t;
}

struct Ray {
	//Destination = origin + t*direction
	Vector3f origin;
	Vector3f direction, direction_inv;
	double t;//transportation time,
	double t_min, t_max;

	Ray(const Vector3f& ori, const Vector3f& dir, const double _t = 0.0) : origin(ori), direction(dir), t(_t) {
		direction_inv = Vector3f(1. / direction.x, 1. / direction.y, 1. / direction.z);
		t_min = 0.0;
		t_max = std::numeric_limits<double>::max();
	}

	Vector3f operator()(double t) const { return origin + direction * t; }

	friend std::ostream &operator<<(std::ostream& os, const Ray& r) {
		os << "[origin:=" << r.origin << ", direction=" << r.direction << ", time=" << r.t << "]\n";
		return os;
	}
};

// Bounds Declarations
template <typename T>
class Bounds2 {
  public:
    // Bounds2 Public Methods
    Bounds2() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Vector2<T>(maxNum, maxNum);
        pMax = Vector2<T>(minNum, minNum);
    }
    explicit Bounds2(const Vector2<T> &p) : pMin(p), pMax(p) {}
    Bounds2(const Vector2<T> &p1, const Vector2<T> &p2) {
        pMin = Vector2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
        pMax = Vector2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
    }
    template <typename U>
    explicit operator Bounds2<U>() const {
        return Bounds2<U>((Vector2<U>)pMin, (Vector2<U>)pMax);
    }
	Vector2f Centroid() { return 0.5 * pMin + 0.5 * pMax; }
    Vector2<T> Diagonal() const { return pMax - pMin; }
    T Area() const {
        Vector2<T> d = pMax - pMin;
        return (d.x * d.y);
    }
    int MaximumExtent() const {
        Vector2<T> diag = Diagonal();
        if (diag.x > diag.y)
            return 0;
        else
            return 1;
    }
    inline const Vector2<T> &operator[](int i) const {
        assert(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    inline Vector2<T> &operator[](int i) {
        assert(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    bool operator==(const Bounds2<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    bool operator!=(const Bounds2<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    Vector2<T> Lerp(const Vector2f &t) const {
        return Vector2<T>(Lerp(t.x, pMin.x, pMax.x),
                         Lerp(t.y, pMin.y, pMax.y));
    }
    Vector2<T> Offset(const Vector2<T> &p) const {
        Vector2<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        return o;
    }
    void BoundingSphere(Vector2<T> *c, float *rad) const {
        *c = (pMin + pMax) / 2;
        *rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
    }
    friend std::ostream &operator<<(std::ostream &os, const Bounds2<T> &b) {
        os << "[ " << b.pMin << " - " << b.pMax << " ]";
        return os;
    }

    // Bounds2 Public Data
    Vector2<T> pMin, pMax;
};

template <typename T>
class Bounds3 {
  public:
    // Bounds3 Public Methods
    Bounds3() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Vector3<T>(maxNum, maxNum, maxNum);
        pMax = Vector3<T>(minNum, minNum, minNum);
    }
    explicit Bounds3(const Vector3<T> &p) : pMin(p), pMax(p) {}
    Bounds3(const Vector3<T> &p1, const Vector3<T> &p2)
        : pMin(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
               std::min(p1.z, p2.z)),
          pMax(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
               std::max(p1.z, p2.z)) {}
    const Vector3<T> &operator[](int i) const;
    Vector3<T> &operator[](int i);
    bool operator==(const Bounds3<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    bool operator!=(const Bounds3<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
	Vector3f Centroid() { return 0.5 * pMin + 0.5 * pMax; }
    Vector3<T> Corner(int corner) const {
        assert(corner >= 0 && corner < 8);
        return Vector3<T>((*this)[(corner & 1)].x,
                         (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }
    Vector3<T> Diagonal() const { return pMax - pMin; }
    T SurfaceArea() const {
        Vector3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    T Volume() const {
        Vector3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }
    int MaximumExtent() const {
        Vector3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
    Vector3<T> Lerp(const Vector3f &t) const {
        return Vector3<T>(Lerp(t.x, pMin.x, pMax.x),
                         Lerp(t.y, pMin.y, pMax.y),
                         Lerp(t.z, pMin.z, pMax.z));
    }
    Vector3<T> Offset(const Vector3<T> &p) const {
        Vector3<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
    void BoundingSphere(Vector3<T> *center, float *radius) const {
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }
    template <typename U>
    explicit operator Bounds3<U>() const {
        return Bounds3<U>((Vector3<U>)pMin, (Vector3<U>)pMax);
    }
    inline bool IntersectP(const Ray &ray, const Vector3f &invDir,
						   const std::array<int, 3>& dirIsNeg) const;
    friend std::ostream &operator<<(std::ostream &os, const Bounds3<T> &b) {
        os << "[ " << b.pMin << " - " << b.pMax << " ]";
        return os;
    }

    // Bounds3 Public Data
    Vector3<T> pMin, pMax;
};

typedef Bounds2<float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds3<float> Bounds3f;
typedef Bounds3<int> Bounds3i;

class Bounds2iIterator : public std::forward_iterator_tag {
  public:
    Bounds2iIterator(const Bounds2i &b, const Vector2i &pt)
        : p(pt), bounds(&b) {}
    Bounds2iIterator operator++() {
        advance();
        return *this;
    }
    Bounds2iIterator operator++(int) {
        Bounds2iIterator old = *this;
        advance();
        return old;
    }
    bool operator==(const Bounds2iIterator &bi) const {
        return p == bi.p && bounds == bi.bounds;
    }
    bool operator!=(const Bounds2iIterator &bi) const {
        return p != bi.p || bounds != bi.bounds;
    }

    Vector2i operator*() const { return p; }

  private:
    void advance() {
        ++p.x;
        if (p.x == bounds->pMax.x) {
            p.x = bounds->pMin.x;
            ++p.y;
        }
    }
    Vector2i p;
    const Bounds2i *bounds;
};

// Geometry Inline Functions
//template <typename T>
//inline Vector3<T>::Vector3(const Vector3<T> &p)
//    : x(p.x), y(p.y), z(p.z) {
//    assert(!HasNaNs());
//}

template <typename T, typename U>
inline Vector3<T> operator*(U s, const Vector3<T> &v) {
    return v * s;
}
template <typename T>
Vector3<T> Abs(const Vector3<T> &v) {
    return Vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

template <typename T>
inline T Dot(const Vector3<T> &v1, const Vector3<T> &v2) {
    assert(!v1.HasNaNs() && !v2.HasNaNs());
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T>
inline T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) {
    assert(!v1.HasNaNs() && !v2.HasNaNs());
    return std::abs(Dot(v1, v2));
}

template <typename T>
inline Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
    assert(!v1.HasNaNs() && !v2.HasNaNs());
    double v1x = v1.x, v1y = v1.y, v1z = v1.z;
    double v2x = v2.x, v2y = v2.y, v2z = v2.z;
    return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
                      (v1x * v2y) - (v1y * v2x));
}

template <typename T>
inline Vector3<T> Normalize(const Vector3<T> &v) {
    return v / v.Length();
}
template <typename T>
T MinComponent(const Vector3<T> &v) {
    return std::min(v.x, std::min(v.y, v.z));
}

template <typename T>
T MaxComponent(const Vector3<T> &v) {
    return std::max(v.x, std::max(v.y, v.z));
}

template <typename T>
int MaxDimension(const Vector3<T> &v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

template <typename T>
Vector3<T> Min(const Vector3<T> &p1, const Vector3<T> &p2) {
    return Vector3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                      std::min(p1.z, p2.z));
}

template <typename T>
Vector3<T> Max(const Vector3<T> &p1, const Vector3<T> &p2) {
    return Vector3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                      std::max(p1.z, p2.z));
}

template <typename T>
Vector3<T> Permute(const Vector3<T> &v, int x, int y, int z) {
    return Vector3<T>(v[x], v[y], v[z]);
}

template <typename T>
inline void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2,
                             Vector3<T> *v3) {
    if (std::abs(v1.x) > std::abs(v1.y))
        *v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        *v2 = Vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
    *v3 = Cross(v1, *v2);
}

template <typename T, typename U>
inline Vector2<T> operator*(U f, const Vector2<T> &v) {
    return v * f;
}
template <typename T>
inline float Dot(const Vector2<T> &v1, const Vector2<T> &v2) {
    assert(!v1.HasNaNs() && !v2.HasNaNs());
    return v1.x * v2.x + v1.y * v2.y;
}

template <typename T>
inline float AbsDot(const Vector2<T> &v1, const Vector2<T> &v2) {
    assert(!v1.HasNaNs() && !v2.HasNaNs());
    return std::abs(Dot(v1, v2));
}

template <typename T>
inline Vector2<T> Normalize(const Vector2<T> &v) {
    return v / v.Length();
}
template <typename T>
Vector2<T> Abs(const Vector2<T> &v) {
    return Vector2<T>(std::abs(v.x), std::abs(v.y));
}

template <typename T>
inline float Distance(const Vector3<T> &p1, const Vector3<T> &p2) {
    return (p1 - p2).Length();
}

template <typename T>
inline float DistanceSquared(const Vector3<T> &p1, const Vector3<T> &p2) {
    return (p1 - p2).LengthSquared();
}

template <typename T>
Vector3<T> Lerp(const Vector3<T> &p0, const Vector3<T> &p1, float t) {
    return (1 - t) * p0 + t * p1;
}

template <typename T>
Vector3<T> Floor(const Vector3<T> &p) {
    return Vector3<T>(std::floor(p.x), std::floor(p.y), std::floor(p.z));
}

template <typename T>
Vector3<T> Ceil(const Vector3<T> &p) {
    return Vector3<T>(std::ceil(p.x), std::ceil(p.y), std::ceil(p.z));
}

template <typename T>
inline float Distance(const Vector2<T> &p1, const Vector2<T> &p2) {
    return (p1 - p2).Length();
}

template <typename T>
inline float DistanceSquared(const Vector2<T> &p1, const Vector2<T> &p2) {
    return (p1 - p2).LengthSquared();
}

template <typename T>
Vector2<T> Floor(const Vector2<T> &p) {
    return Vector2<T>(std::floor(p.x), std::floor(p.y));
}

template <typename T>
Vector2<T> Ceil(const Vector2<T> &p) {
    return Vector2<T>(std::ceil(p.x), std::ceil(p.y));
}

template <typename T>
Vector2<T> Lerp(const Vector2<T> &v0, const Vector2<T> &v1, float t) {
    return (1 - t) * v0 + t * v1;
}

template <typename T>
Vector2<T> Min(const Vector2<T> &pa, const Vector2<T> &pb) {
    return Vector2<T>(std::min(pa.x, pb.x), std::min(pa.y, pb.y));
}

template <typename T>
Vector2<T> Max(const Vector2<T> &pa, const Vector2<T> &pb) {
    return Vector2<T>(std::max(pa.x, pb.x), std::max(pa.y, pb.y));
}

template <typename T>
inline const Vector3<T> &Bounds3<T>::operator[](int i) const {
    assert(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T>
inline Vector3<T> &Bounds3<T>::operator[](int i) {
    assert(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T>
Bounds3<T> Union(const Bounds3<T> &b, const Vector3<T> &p) {
    Bounds3<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T>
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    Bounds3<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    // Important: assign to pMin/pMax directly and don't run the Bounds2()
    // constructor, since it takes min/max of the points passed to it.  In
    // turn, that breaks returning an invalid bound for the case where we
    // intersect non-overlapping bounds (as we'd like to happen).
    Bounds3<T> ret;
    ret.pMin = Max(b1.pMin, b2.pMin);
    ret.pMax = Min(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T>
bool Inside(const Vector3<T> &p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template <typename T>
bool InsideExclusive(const Vector3<T> &p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
            p.y < b.pMax.y && p.z >= b.pMin.z && p.z < b.pMax.z);
}

template <typename T, typename U>
inline Bounds3<T> Expand(const Bounds3<T> &b, U delta) {
    return Bounds3<T>(b.pMin - Vector3<T>(delta, delta, delta),
                      b.pMax + Vector3<T>(delta, delta, delta));
}

// Minimum squared distance from point to box; returns zero if point is
// inside.
template <typename T, typename U>
inline float DistanceSquared(const Vector3<T> &p, const Bounds3<U> &b) {
    float dx = std::max({float(0), b.pMin.x - p.x, p.x - b.pMax.x});
    float dy = std::max({float(0), b.pMin.y - p.y, p.y - b.pMax.y});
    float dz = std::max({float(0), b.pMin.z - p.z, p.z - b.pMax.z});
    return dx * dx + dy * dy + dz * dz;
}

template <typename T, typename U>
inline float Distance(const Vector3<T> &p, const Bounds3<U> &b) {
    return std::sqrt(DistanceSquared(p, b));
}

inline Bounds2iIterator begin(const Bounds2i &b) {
    return Bounds2iIterator(b, b.pMin);
}

inline Bounds2iIterator end(const Bounds2i &b) {
    // Normally, the ending point is at the minimum x value and one past
    // the last valid y value.
    Vector2i pEnd(b.pMin.x, b.pMax.y);
    // However, if the bounds are degenerate, override the end point to
    // equal the start point so that any attempt to iterate over the bounds
    // exits out immediately.
    if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y)
        pEnd = b.pMin;
    return Bounds2iIterator(b, pEnd);
}

template <typename T>
Bounds2<T> Union(const Bounds2<T> &b, const Vector2<T> &p) {
    Bounds2<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T>
Bounds2<T> Union(const Bounds2<T> &b, const Bounds2<T> &b2) {
    Bounds2<T> ret;
    ret.pMin = Min(b.pMin, b2.pMin);
    ret.pMax = Max(b.pMax, b2.pMax);
    return ret;
}

template <typename T>
Bounds2<T> Intersect(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Important: assign to pMin/pMax directly and don't run the Bounds2()
    // constructor, since it takes min/max of the points passed to it.  In
    // turn, that breaks returning an invalid bound for the case where we
    // intersect non-overlapping bounds (as we'd like to happen).
    Bounds2<T> ret;
    ret.pMin = Max(b1.pMin, b2.pMin);
    ret.pMax = Min(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
bool Overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
    bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
    return (x && y);
}

template <typename T>
bool Inside(const Vector2<T> &pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x <= b.pMax.x && pt.y >= b.pMin.y &&
            pt.y <= b.pMax.y);
}

template <typename T>
bool InsideExclusive(const Vector2<T> &pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x < b.pMax.x && pt.y >= b.pMin.y &&
            pt.y < b.pMax.y);
}

template <typename T, typename U>
Bounds2<T> Expand(const Bounds2<T> &b, U delta) {
    return Bounds2<T>(b.pMin - Vector2<T>(delta, delta),
                      b.pMax + Vector2<T>(delta, delta));
}

template <typename T>
inline bool Bounds3<T>::IntersectP(const Ray &ray, const Vector3f &invDir, 
								   const std::array<int, 3>& dirIsNeg) const {
	// invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
	// dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
	// TODO test if ray bound intersects

	Vector3f t_1 = (pMin - ray.origin) * invDir;
	Vector3f t_2 = (pMax - ray.origin) * invDir;

	if (!dirIsNeg[0])
	{
		float tmp = t_1.x;
		t_1.x = t_2.x;
		t_2.x = tmp;
	}

	if (!dirIsNeg[1])
	{
		float tmp = t_1.y;
		t_1.y = t_2.y;
		t_2.y = tmp;
	}

	if (!dirIsNeg[2])
	{
		float tmp = t_1.z;
		t_1.z = t_2.z;
		t_2.z = tmp;
	}

	float t_enter = std::max(std::max(t_1.x, t_1.y), t_1.z);
	float t_exit = std::min(std::min(t_2.x, t_2.y), t_2.z);

	if (t_enter <= t_exit && t_exit > 0)
		return true;
	else
		return false;
}

//inline Vector3f OffsetRayOrigin(const Vector3f &p, const Vector3f &pError,
//                               const Normal3f &n, const Vector3f &w) {
//    float d = Dot(Abs(n), pError);
//    Vector3f offset = d * Vector3f(n);
//    if (Dot(w, n) < 0) offset = -offset;
//    Vector3f po = p + offset;
//    // Round offset point _po_ away from _p_
//    for (int i = 0; i < 3; ++i) {
//        if (offset[i] > 0)
//            po[i] = NextfloatUp(po[i]);
//        else if (offset[i] < 0)
//            po[i] = NextfloatDown(po[i]);
//    }
//    return po;
//}

inline Vector3f SphericalDirection(float sinTheta, float cosTheta, float phi) {
    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                    cosTheta);
}

inline Vector3f SphericalDirection(float sinTheta, float cosTheta, float phi,
                                   const Vector3f &x, const Vector3f &y,
                                   const Vector3f &z) {
    return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y +
           cosTheta * z;
}

inline float SphericalTheta(const Vector3f &v) {
    return std::acos(Clamp(v.z, -1, 1));
}

inline float SphericalPhi(const Vector3f &v) {
    float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * M_PI) : p;
}
