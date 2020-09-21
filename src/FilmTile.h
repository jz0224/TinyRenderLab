#pragma once
#include "Bounds.h"
#include <vector>

class FilmTile
{
public:
	FilmTile(const Bounds2& frame) : frame(frame),
		pixels(frame.Diagonal().x, std::vector<Vector3f>(frame.Diagonal().y)) { }

	void AddSample(const Vector2f& pos, const Vector3f& radiance);
	const std::vector<Vector2f> AllPos() const;
	const Vector3f& At(const Vector2f& pos) const;

private:
	const Bounds2 frame;
	std::vector<std::vector<Vector3f>> pixels;
};