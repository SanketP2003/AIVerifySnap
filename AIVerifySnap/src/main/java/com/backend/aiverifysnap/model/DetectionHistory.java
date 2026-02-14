package com.backend.aiverifysnap.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
@Table(name = "detection_history")
public class DetectionHistory {

    @Id
    @Column(name = "scan_id")
    private Long scanId;

    @Column(name = "image_path")
    private String imagePath;

    @Column(name = "result_label")
    private String resultLabel;

    @Column(name = "confidence_Score")
    private Double confidenceScore;

    @Column(name = "analysis_metadata")
    private String analysisMetadata;

    @Column(name = "scan_timestamp")
    private LocalDateTime scanTimestamp;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private Users id;
}
