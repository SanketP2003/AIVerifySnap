import { ImageComparisonSlider } from "./ImageComparisonSlider";
import { Info, Sparkles } from "lucide-react";

interface HeatmapViewerProps {
    originalImage: string;
    heatmapImage: string;
    title?: string;
    subtitle?: string;
}

export function HeatmapViewer({
    originalImage,
    heatmapImage,
    title = "Grad-CAM Heatmap",
    subtitle = "Original vs AI Visualization",
}: HeatmapViewerProps) {
    return (
        <div className="flex flex-col space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 px-1">
                <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    <h3 className="font-semibold text-lg tracking-tight">{title}</h3>
                </div>
                <p className="text-xs sm:text-sm text-muted-foreground">{subtitle}</p>
            </div>

            <ImageComparisonSlider
                originalImage={originalImage}
                comparisonImage={heatmapImage}
                originalLabel="Original"
                comparisonLabel="Forensic Heatmap"
            />

            {/* Gradient scale legend */}
            <div className="flex flex-col sm:flex-row justify-between items-center gap-2 px-4 py-3 glass-panel rounded-2xl text-xs font-medium text-muted-foreground">
                <div className="flex items-center gap-1.5 shrink-0">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    <span>Low Attention (Cold)</span>
                </div>
                <div className="w-full sm:flex-1 h-2.5 mx-0 sm:mx-4 rounded-full bg-gradient-to-r from-blue-500 via-green-400 to-red-500 max-w-sm" />
                <div className="flex items-center gap-1.5 shrink-0">
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    <span>High Attention (Hot)</span>
                </div>
            </div>

            <p className="text-xs text-muted-foreground/80 flex items-center gap-1.5 px-2">
                <Info className="w-3.5 h-3.5 shrink-0" />
                Drag the divider handle to inspect spatial compression anomalies and model attention hotspots.
            </p>
        </div>
    );
}
