#pragma once

#include <memory>
#include "Bounds.h"
#include "Scene.h"
#include <string>

class FilmTile;

class Film
{
public:
	Film(const Scene& scene, const std::string filename)
		: width(scene.width), height(scene.height),
		pixels(width, std::vector<Vector3f>(height)),
		m_FileName(filename) {};

	const std::shared_ptr<FilmTile> GenFilmTile(const Bounds2& frame) const;
	void MergeFilmTile(std::shared_ptr<FilmTile> filmTile);
	void WriteImage();

private:
	int width;
	int height;
	std::vector<std::vector<Vector3f>> pixels;
	std::string m_FileName;
};