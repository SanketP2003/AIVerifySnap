package com.backend.aiverifysnap.repository;

import com.backend.aiverifysnap.model.DetectionHistory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface DetectionRepository extends JpaRepository<DetectionHistory, Long> {
    long countByResultLabel(String resultLabel);
    List<DetectionHistory> findAllByOrderByScanTimestampDesc();
}
