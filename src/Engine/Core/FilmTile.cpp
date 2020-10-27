#include "Core/FilmTile.h"

void FilmTile::AddSample(const Vector2f& pos, const Vector3f& radiance)
{
	pixels[pos.x - frame.pMin.x][pos.y - frame.pMin.y] = radiance;
}

const std::vector<Vector2f> FilmTile::AllPos() const
{
	std::vector<Vector2f> posArr;
	posArr.reserve(frame.Area());
	for (int x = frame.pMin.x; x < frame.pMax.x; x++)
	{
		for (int y = frame.pMin.y; y < frame.pMax.y; y++)
		{
			posArr.push_back(Vector2f(x, y));
		}
	}
	return posArr;
}

const Vector3f& FilmTile::At(const Vector2f& pos) const
{
	return pixels[pos.x - frame.pMin.x][pos.y - frame.pMin.y];
}