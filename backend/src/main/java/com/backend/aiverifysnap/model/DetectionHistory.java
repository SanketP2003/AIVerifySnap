package com.backend.aiverifysnap.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.persistence.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "detection_history")
public class DetectionHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "detection_history_seq")
    @SequenceGenerator(name = "detection_history_seq", sequenceName = "detection_history_scan_id_seq", allocationSize = 1)
    @Column(name = "scan_id")
    private Long scanId;

    @Column(name = "image_path")
    private String imagePath;

    @Column(name = "result_label")
    private String resultLabel;

    @Column(name = "confidence_Score")
    private Double confidenceScore;

    @Column(name = "analysis_metadata", columnDefinition = "TEXT")
    private String analysisMetadata;

    @Column(name = "scan_timestamp")
    private LocalDateTime scanTimestamp;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id")
    @JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
    private Users user;

    public DetectionHistory() {}

    public Long getScanId() {
        return scanId;
    }

    public void setScanId(Long scanId) {
        this.scanId = scanId;
    }

    public String getImagePath() {
        return imagePath;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }

    public String getResultLabel() {
        return resultLabel;
    }

    public void setResultLabel(String resultLabel) {
        this.resultLabel = resultLabel;
    }

    public Double getConfidenceScore() {
        return confidenceScore;
    }

    public void setConfidenceScore(Double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }

    public String getAnalysisMetadata() {
        return analysisMetadata;
    }

    public void setAnalysisMetadata(String analysisMetadata) {
        this.analysisMetadata = analysisMetadata;
    }

    public LocalDateTime getScanTimestamp() {
        return scanTimestamp;
    }

    public void setScanTimestamp(LocalDateTime scanTimestamp) {
        this.scanTimestamp = scanTimestamp;
    }

    public Users getUser() {
        return user;
    }

    public void setUser(Users user) {
        this.user = user;
    }

}