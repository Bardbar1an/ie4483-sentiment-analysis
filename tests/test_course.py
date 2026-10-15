"""Course-specific submission contract checks."""
import shutil, unittest, uuid
from pathlib import Path
import pandas as pd
from train_course import export_course_submission, scores

class CourseTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parent
        self.folder=self.root/('.course-test-'+uuid.uuid4().hex)
        self.folder.mkdir()
        self.raw=[{'reviews':'Great, "really" great!\nWorks well.'},{'reviews':'Bad product.'}]
        self.template=self.folder/'template.csv'
        pd.DataFrame({'reviews':[r['reviews'] for r in self.raw],'predictions':[0,0]}).to_csv(self.template,index=False)
    def tearDown(self):
        assert self.folder.resolve().parent==self.root
        shutil.rmtree(self.folder)
    def test_course_export_preserves_exact_text_and_order(self):
        out=self.folder/'submission.csv'
        export_course_submission(self.raw,self.template,[1,0],out)
        actual=pd.read_csv(out,keep_default_na=False)
        self.assertEqual(list(actual.columns),['reviews','predictions'])
        self.assertEqual(actual.reviews.tolist(),[r['reviews'] for r in self.raw])
        self.assertEqual(actual.predictions.tolist(),[1,0])
    def test_reordered_template_is_rejected(self):
        frame=pd.read_csv(self.template); frame.iloc[::-1].to_csv(self.template,index=False)
        with self.assertRaisesRegex(ValueError,'exactly match'):
            export_course_submission(self.raw,self.template,[1,0],self.folder/'wrong.csv')
    def test_invalid_predictions_and_overwrite_are_rejected(self):
        out=self.folder/'submission.csv'
        for labels in [[1],[1,2]]:
            with self.assertRaisesRegex(ValueError,'binary prediction'):
                export_course_submission(self.raw,self.template,labels,out)
        export_course_submission(self.raw,self.template,[1,0],out)
        with self.assertRaisesRegex(ValueError,'already exists'):
            export_course_submission(self.raw,self.template,[0,1],out)
    def test_majority_accuracy_does_not_hide_zero_negative_recall(self):
        result=scores([1]*9+[0],[1]*10)
        self.assertAlmostEqual(result['accuracy'],.9)
        self.assertEqual(result['negative_recall'],0)
        self.assertLess(result['macro_f1'],.5)

if __name__=='__main__': unittest.main()
