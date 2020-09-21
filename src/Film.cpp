#include "Film.h"
#include "FilmTile.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const std::shared_ptr<FilmTile> Film::GenFilmTile(const Bounds2 & frame) const 
{
	return std::make_shared<FilmTile>(frame);
}

void Film::MergeFilmTile(std::shared_ptr<FilmTile> filmTile)
{
	for (const auto pos : filmTile->AllPos()) 
	{
		pixels[pos.x][pos.y] += filmTile->At(pos);
	}
}

void Film::WriteImage()
{
	std::vector<unsigned char> normalizedBuffer;
	normalizedBuffer.reserve(width * height * 3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			static unsigned char color[3];
			color[0] = (unsigned char)(255 * std::pow(std::clamp(pixels[j][i].x, 0.f, 1.f), 0.6f));
			color[1] = (unsigned char)(255 * std::pow(std::clamp(pixels[j][i].y, 0.f, 1.f), 0.6f));
			color[2] = (unsigned char)(255 * std::pow(std::clamp(pixels[j][i].z, 0.f, 1.f), 0.6f));
			normalizedBuffer.insert(normalizedBuffer.end(), color, color + 3);
		}
	}

	stbi_write_jpg(m_FileName.c_str(), width, height, 3, &normalizedBuffer[0], 100);
}