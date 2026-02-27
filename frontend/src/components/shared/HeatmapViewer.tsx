import { ImageComparisonSlider } from "./ImageComparisonSlider";

interface HeatmapViewerProps {
    originalImage: string;
    heatmapImage: string;
}

export function HeatmapViewer({ originalImage, heatmapImage }: HeatmapViewerProps) {
    return (
        <div className="flex flex-col space-y-4">
            <div className="flex items-center justify-between px-2">
                <h3 className="font-semibold text-lg tracking-tight">Grad-CAM Heatmap</h3>
                <p className="text-sm text-balance text-muted-foreground">Original vs AI Visualization</p>
            </div>
            <ImageComparisonSlider originalImage={originalImage} comparisonImage={heatmapImage} />
            <div className="flex justify-between items-center px-4 py-2 mt-4 glass-panel rounded-full text-xs font-medium text-muted-foreground">
                <span>Low attention (Blue)</span>
                <div className="flex-1 h-3 mx-4 rounded bg-gradient-to-r from-blue-500 via-green-400 to-red-500 max-w-sm"></div>
                <span>High attention (Red)</span>
            </div>
        </div>
    );
}
