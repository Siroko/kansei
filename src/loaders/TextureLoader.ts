/**
 * Class responsible for loading textures from a given URL.
 */
class TextureLoader {
    private imageBitmap?: ImageBitmap;

    /**
     * Loads an image from the specified URL and returns an ImageBitmap.
     * 
     * @param url The URL of the image to load.
     * @returns A promise that resolves to an ImageBitmap of the loaded image.
     * @throws Will throw an error if the image cannot be fetched or processed.
     */
    public async load(url: string): Promise<ImageBitmap> {
        const response = await fetch(url);
        const blob = await response.blob();
        this.imageBitmap = await createImageBitmap(blob);

        return this.imageBitmap;
    }
}

export { TextureLoader }
