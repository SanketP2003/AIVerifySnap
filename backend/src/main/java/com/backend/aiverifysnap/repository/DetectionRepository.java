package com.backend.aiverifysnap.repository;

import com.backend.aiverifysnap.model.DetectionHistory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDateTime;
import java.util.List;

public interface DetectionRepository extends JpaRepository<DetectionHistory, Long> {
    List<DetectionHistory> findByUserIdOrderByScanTimestampDesc(Long userId);

    List<DetectionHistory> findByScanTimestampAfterOrderByScanTimestampDesc(LocalDateTime cutoff);

    List<DetectionHistory> findByUserIdAndScanTimestampAfterOrderByScanTimestampDesc(Long userId, LocalDateTime cutoff);
}
